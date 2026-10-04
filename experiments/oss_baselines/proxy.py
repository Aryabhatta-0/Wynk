"""Metering pass-through proxy for an OpenAI-compatible backend.

Every system in the comparison (Wynk AND every external framework) talks to the model through
this proxy, so LLM calls, backend-reported token usage, backend-reported cost, latency and the
request parameters actually sent (temperature, max_tokens, tools, ...) are measured the same way
for all of them instead of trusting each framework's own accounting.

    base_url for one run = http://127.0.0.1:<port>/r/<run_id>/v1

The proxy does not change what the model sees. It (1) swaps the client's placeholder key for the
real key, so framework code never holds it, and (2) on streamed requests sets
``stream_options.include_usage`` so the backend reports usage at all. It never retries: each
client keeps its own (equal) retry policy.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

_UPSTREAM_TIMEOUT_S = 300.0


@dataclass
class CallRecord:
    """One proxied request (raw bodies kept for auditability)."""

    run_id: str
    index: int
    path: str
    started_at: float
    latency_s: float = 0.0
    status: int | None = None
    stream: bool = False
    model: str | None = None
    temperature: Any = None
    max_tokens: Any = None
    seed: Any = None
    tools: list[str] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    error: str | None = None
    request_body: Any = None
    response_body: Any = None

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def summarize_calls(calls: list[CallRecord]) -> dict[str, Any]:
    """Aggregate telemetry. Token/cost totals are ``None`` when any chat call lacked usage, so a
    partial number is never reported as the total."""
    chat = [c for c in calls if c.path.endswith("chat/completions")]

    def total(attr: str) -> float | None:
        vals = [getattr(c, attr) for c in chat]
        return None if any(v is None for v in vals) else sum(vals)

    prompt, completion = total("prompt_tokens"), total("completion_tokens")
    return {
        "llm_calls": len(chat),
        "llm_calls_ok": sum(1 for c in chat if c.status == 200),
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": None if prompt is None or completion is None else prompt + completion,
        "cost_usd": total("cost_usd"),
        # lower bound used for cap checks when some call has no usage (e.g. client gave up)
        "known_total_tokens": sum(
            (c.prompt_tokens or 0) + (c.completion_tokens or 0) for c in chat
        ),
        "llm_time_s": round(sum(c.latency_s for c in chat), 3),
    }


def _usage(obj: Any) -> tuple[int | None, int | None, float | None]:
    usage = obj.get("usage") if isinstance(obj, dict) else None
    if not isinstance(usage, dict):
        return None, None, None
    cost = usage.get("cost")
    try:
        cost_f = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        cost_f = None
    return usage.get("prompt_tokens"), usage.get("completion_tokens"), cost_f


class MeteringProxy:
    def __init__(self, upstream_base_url: str, api_key: str | None) -> None:
        self.upstream = upstream_base_url.rstrip("/")
        self.api_key = api_key
        self._lock = threading.Lock()
        self._calls: dict[str, list[CallRecord]] = {}
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # --- lifecycle -------------------------------------------------------------------------
    def start(self) -> MeteringProxy:
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                proxy._handle(self, "POST")

            def do_GET(self) -> None:  # noqa: N802
                proxy._handle(self, "GET")

            def log_message(self, *args: Any) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    @property
    def port(self) -> int:
        assert self._server is not None
        return self._server.server_address[1]

    def base_url(self, run_id: str) -> str:
        return f"http://127.0.0.1:{self.port}/r/{run_id}/v1"

    def calls(self, run_id: str) -> list[CallRecord]:
        with self._lock:
            return list(self._calls.get(run_id, []))

    # --- request handling ------------------------------------------------------------------
    def _record(self, run_id: str, path: str) -> CallRecord:
        with self._lock:
            calls = self._calls.setdefault(run_id, [])
            rec = CallRecord(run_id=run_id, index=len(calls), path=path, started_at=time.time())
            calls.append(rec)
            return rec

    def _handle(self, h: BaseHTTPRequestHandler, method: str) -> None:
        parts = h.path.split("/", 4)  # ['', 'r', run_id, 'v1', rest]
        if len(parts) < 5 or parts[1] != "r" or parts[3] != "v1":
            h.send_error(404, "expected /r/<run_id>/v1/...")
            return
        run_id, rest = parts[2], parts[4]
        rec = self._record(run_id, rest)
        t0 = time.perf_counter()
        body: bytes | None = None
        if method == "POST":
            body = h.rfile.read(int(h.headers.get("Content-Length") or 0))
            try:
                payload = json.loads(body or b"{}")
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                rec.stream = bool(payload.get("stream"))
                rec.model = payload.get("model")
                rec.temperature = payload.get("temperature")
                rec.max_tokens = payload.get("max_tokens", payload.get("max_completion_tokens"))
                rec.seed = payload.get("seed")
                rec.tools = [
                    (t.get("function") or {}).get("name", "?")
                    for t in payload.get("tools") or []
                    if isinstance(t, dict)
                ]
                if rec.stream:
                    opts = dict(payload.get("stream_options") or {})
                    opts["include_usage"] = True
                    payload["stream_options"] = opts
                    body = json.dumps(payload).encode()
                rec.request_body = payload
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.upstream}/{rest}", data=body, headers=headers, method=method
        )
        try:
            resp = urllib.request.urlopen(req, timeout=_UPSTREAM_TIMEOUT_S)  # noqa: S310
        except urllib.error.HTTPError as exc:
            data = exc.read()
            rec.status, rec.error = exc.code, data.decode("utf-8", "replace")[:2000]
            rec.latency_s = time.perf_counter() - t0
            self._reply(h, exc.code, exc.headers.get("Content-Type"), data, exc.headers)
            return
        except Exception as exc:  # connection failures -> 502 to the client
            rec.status, rec.error = 502, f"{type(exc).__name__}: {exc}"
            rec.latency_s = time.perf_counter() - t0
            self._reply(h, 502, "application/json", json.dumps({"error": rec.error}).encode())
            return
        with resp:
            rec.status = resp.status
            ctype = resp.headers.get("Content-Type", "application/json")
            if rec.stream and "text/event-stream" in ctype:
                self._relay_stream(h, resp, ctype, rec)
            else:
                data = resp.read()
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    obj = data.decode("utf-8", "replace")
                rec.response_body = obj
                rec.prompt_tokens, rec.completion_tokens, rec.cost_usd = _usage(obj)
                self._reply(h, resp.status, ctype, data)
        rec.latency_s = time.perf_counter() - t0

    @staticmethod
    def _reply(h, status: int, ctype: str | None, data: bytes, src_headers=None) -> None:
        h.send_response(status)
        h.send_header("Content-Type", ctype or "application/json")
        h.send_header("Content-Length", str(len(data)))
        if src_headers is not None and src_headers.get("Retry-After"):
            h.send_header("Retry-After", src_headers.get("Retry-After"))
        h.end_headers()
        try:
            h.wfile.write(data)
        except ConnectionError:  # client timed out and left; its retry is a separate call
            h.close_connection = True

    @staticmethod
    def _relay_stream(h, resp, ctype: str, rec: CallRecord) -> None:
        h.send_response(rec.status or 200)
        h.send_header("Content-Type", ctype)
        h.send_header("Cache-Control", "no-cache")
        h.send_header("Connection", "close")
        h.end_headers()
        text_parts: list[str] = []
        tool_chunks: list[Any] = []
        for raw in resp:
            h.wfile.write(raw)
            h.wfile.flush()
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            p, c, cost = _usage(chunk)
            if p is not None:
                rec.prompt_tokens, rec.completion_tokens, rec.cost_usd = p, c, cost
            for choice in chunk.get("choices") or []:
                delta = choice.get("delta") or {}
                if delta.get("content"):
                    text_parts.append(delta["content"])
                if delta.get("tool_calls"):
                    tool_chunks.extend(delta["tool_calls"])
        rec.response_body = {"streamed_text": "".join(text_parts), "tool_call_deltas": tool_chunks}
        h.close_connection = True
