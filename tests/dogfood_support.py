"""v0.1 dogfood support: a fresh user's dataset through the REAL product wiring.

Nothing here replaces a product layer. The API is ``api.product.build_api`` with the same
``registry_runtime`` / ``registry_inference`` factories ``api_from_args`` installs, a registry
FILE (re-read on every inference), and the real ``OpenAICompatibleClient`` talking real HTTP.
Only the far end of that HTTP connection is fake: ``FakeProvider`` is a local
OpenAI-compatible server that answers the order-desk prompts deterministically, so the tests
never send data to a hosted provider and need no credentials.
"""

from __future__ import annotations

import io
import json
import re
import threading
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from api.product import ERROR_STATUS, build_api, registry_inference, registry_runtime
from api.tenancy import TenantRouter
from core.models import ModelCapability, ModelEntry, ModelPricing, ModelRegistry
from runtime.backends.brave_search import BraveWebSearchSource, websearch_from_env
from runtime.backends.openai_compatible import ADAPTER, openai_compatible_model_hash

FIXTURE = Path(__file__).parent / "fixtures" / "dogfood" / "order_desk.csv"
DATASET_ID = "order-desk"
TASK_ID = "order-desk"
LINEAGE = f"{DATASET_ID}.{TASK_ID}"
# A fake provider credential. It must never be persisted or returned (asserted by the tests).
PROVIDER_KEY = "sk-dogfood-provider-secret-0123456789abcdef"
MODEL_ID = "dogfood-desk-model"
REVISION = "2026-10-01"
REGISTER = {
    "dataset_id": DATASET_ID,
    "name": "Order desk",
    "input_columns": ["question"],
    "context_columns": ["order_record"],
    "target_columns": ["answer"],
    "row_ids": "column",
    "id_column": "ticket_id",
}
SPLIT_PLAN = {"seed": 11, "validation_bps": 2000, "test_bps": 2000}
INSTRUCTIONS = "Name the destination city of the order, using only the order record."
SHIPS_TO = re.compile(r"ships to ([A-Z][a-z]+)")


# -- the fake far end of the model connection -----------------------------------------------
@dataclass
class FakeProvider:
    """Local OpenAI-compatible chat-completions server. ``status`` != 200 makes it fail every
    request with that HTTP status (503 = provider unavailable)."""

    key: str = PROVIDER_KEY
    status: int = 200
    requests: list[dict[str, Any]] = field(default_factory=list)
    authorizations: list[str | None] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = None

    @property
    def endpoint(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def start(self) -> FakeProvider:
        provider = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                provider.requests.append(body)
                provider.authorizations.append(self.headers.get("Authorization"))
                if provider.status != 200:
                    return self._send(provider.status, {"error": {"message": "unavailable"}})
                if self.headers.get("Authorization") != f"Bearer {provider.key}":
                    return self._send(401, {"error": {"message": "bad key"}})
                content = answer(body["messages"][0]["content"])
                self._send(
                    200,
                    {
                        "choices": [{"message": {"content": json.dumps(content)}}],
                        "usage": {"prompt_tokens": 40, "completion_tokens": 8},
                    },
                )

            def _send(self, status: int, payload: dict[str, Any]) -> None:
                raw = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def stop(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    def prompts(self) -> list[str]:
        return [r["messages"][0]["content"] for r in self.requests]


def answer(prompt: str) -> dict[str, Any]:
    """What a competent model answers: the city after "ships to" in the order record. The
    record is only visible once a GATHER stage put it on a page, so a direct answer is wrong."""
    if prompt.startswith("You extract facts"):
        facts = []
        for block in prompt.split("Pages:")[-1].split("[page ")[1:]:
            page_id, _, text = block.partition("]")
            m = SHIPS_TO.search(text)
            if m:
                facts.append({"field": "answer", "value": m[1], "page_id": page_id, "quote": m[0]})
        return {"facts": facts}
    m = re.search(r'\] answer = "([^"]*)"', prompt)
    value = m[1] if m else ""
    if prompt.startswith("You are given facts"):  # REASON
        return {"facts": [{"field": "answer", "value": value}]}
    if prompt.startswith("Answer the question using ONLY the facts"):  # SYNTHESIZE
        return {"answer": {"answer": value}, "citations": {"answer": [0]}}
    return {"answer": {"answer": "unknown"}}  # DIRECT: no record to read


# -- the fake far end of the websearch connection ------------------------------------------
WEBSEARCH_KEY = "bsk-dogfood-websearch-secret-0123456789"
WEBSEARCH_A = "https://search-a.example.test/res/v1/web/search"
WEBSEARCH_B = "https://search-b.example.test/res/v1/web/search"


@dataclass
class FakeBrave:
    """Transport for the REAL ``BraveWebSearchSource``: answers a query naming an order of the
    fixture with that order's record (a live lookup service), recording every request."""

    status: int = 200
    requests: list[tuple[str, dict[str, str]]] = field(default_factory=list)
    extra: dict[str, str] = field(default_factory=dict)  # question -> record, beyond the fixture

    def __call__(self, url: str, headers: Any, timeout_s: float) -> bytes:
        import urllib.error

        self.requests.append((url, dict(headers)))
        if self.status != 200:
            raise urllib.error.HTTPError(url, self.status, "unavailable", {}, None)  # type: ignore[arg-type]
        records = {row["question"]: row["order_record"] for row in fixture_rows()} | self.extra
        query = next((q for q in records if q.replace(" ", "+").replace("?", "%3F") in url), None)
        results = (
            [
                {
                    "title": "Order record",
                    "url": "https://orders.example.test/r",
                    "description": records[query],
                }
            ]
            if query
            else []
        )
        return json.dumps({"type": "search", "web": {"results": results}}).encode()


def websearch(transport: FakeBrave, endpoint: str = WEBSEARCH_A) -> BraveWebSearchSource:
    """Configured exactly as an operator does (``websearch_from_env``)."""
    source = websearch_from_env(
        {"WYNK_WEBSEARCH_API_KEY": WEBSEARCH_KEY, "WYNK_WEBSEARCH_ENDPOINT": endpoint}, transport
    )
    assert source is not None
    return source


def fixture_rows() -> list[dict[str, str]]:
    import csv

    with FIXTURE.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# -- registry ---------------------------------------------------------------------------------
def entry_for(endpoint: str, **overrides: Any) -> ModelEntry:
    model_id = overrides.pop("model_id", MODEL_ID)
    revision = overrides.pop("revision", REVISION)
    base: dict[str, Any] = dict(
        name="dogfood-desk",
        provider=ADAPTER,
        adapter=ADAPTER,
        model_id=model_id,
        revision=revision,
        endpoint=endpoint,
        model_hash=openai_compatible_model_hash(model_id, revision, endpoint, True),
        capabilities=(ModelCapability.TEXT_GENERATION, ModelCapability.STRUCTURED_OUTPUT),
        pricing=ModelPricing(
            version="dogfood-2026-10",
            currency="USD",
            prompt_per_million=1.0,
            completion_per_million=2.0,
        ),
    )
    base["retry"] = {"max_retries": 0}
    base["timeout"] = {"request_timeout_s": 10.0}
    return ModelEntry.model_validate({**base, **overrides})


def write_registry(path: Path, *entries: ModelEntry) -> None:
    path.write_text(
        ModelRegistry(entries=tuple(entries)).model_dump_json(indent=2), encoding="utf-8"
    )


# -- the product, wired like ``api_from_args`` ------------------------------------------------
def product(
    data_dir: Path,
    registry: Path,
    *,
    key: str | None = PROVIDER_KEY,
    websearch: BraveWebSearchSource | None = None,
) -> TenantRouter:
    return build_api(  # type: ignore[no-any-return]
        data_dir,
        1024 * 1024,
        lambda service: registry_runtime(service, registry, key, websearch),
        registry_inference(registry, key, websearch),
    )


def call(r: Any, key: str | None, method: str, path: str, body: Any = None, **headers: str):
    raw = b""
    h = dict(headers)
    if key is not None:
        h["Authorization"] = f"Bearer {key}"
    if isinstance(body, bytes):
        raw = body
        h.setdefault("Content-Type", "application/octet-stream")
    elif body is not None:
        raw = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    if method in ("POST", "DELETE"):
        h["Content-Length"] = str(len(raw))
    res = r.handle(method, "/api/v1/" + path, h, io.BytesIO(raw))
    if res.raw is not None:  # a download (export): the bytes themselves
        return res.status, res.raw
    if isinstance(res.body, dict):
        json.dumps(res.body, allow_nan=False)
        assert ERROR_STATUS.get(res.body.get("error", {}).get("code"), res.status) == res.status
    return res.status, res.body


def code(result: tuple[int, Any]) -> tuple[int, str]:
    status, body = result
    return status, body["error"]["code"] if isinstance(body, dict) and "error" in body else "ok"


def files_under(root: Path) -> Iterator[Path]:
    return (p for p in root.rglob("*") if p.is_file())
