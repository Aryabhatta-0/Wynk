"""The real HTTP model client against a local OpenAI-compatible fake server, plus the Gemma
stages driven through it. (No external model is contacted.)"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from core.results import FailureKind
from runtime.gemma_client import (
    GemmaConfig,
    GenerationRequest,
    ModelError,
    ModelRole,
    ModelUnavailableError,
    OpenAICompatibleClient,
    client_from_env,
)
from runtime.mvp_genomes import GENOME_A
from runtime.prompts import PROMPT_TEMPLATE_VERSION
from tests.conftest import make_task
from tests.runtime_helpers import PAGES, QUOTE, build_runner, drive, write_snapshot


class FakeBackend:
    """Answers extract/synthesize prompts like a model would; records what it received."""

    def __init__(self, status=200, usage=True):
        self.status, self.usage, self.seen = status, usage, []
        self.fail_first: list[int] = []  # statuses to return before answering normally
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.seen.append((self.path, dict(self.headers), body))
                prompt = body["messages"][0]["content"]
                if "extract facts" in prompt:
                    content = {
                        "facts": [
                            {"field": "capital", "value": "Paris", "page_id": "p1", "quote": QUOTE}
                        ]
                    }
                else:
                    content = {"answer": {"capital": "Paris"}}
                reply = {"choices": [{"message": {"content": json.dumps(content)}}]}
                if outer.usage:
                    reply["usage"] = {"prompt_tokens": 50, "completion_tokens": 10}
                data = json.dumps(reply).encode()
                status = outer.fail_first.pop(0) if outer.fail_first else outer.status
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def backend():
    b = FakeBackend()
    yield b
    b.close()


def req(**kw):
    base = dict(
        role=ModelRole.EXTRACT,
        prompt_template_id="extract.direct",
        prompt_template_version="v",
        input_text="extract facts please",
        output_schema={"type": "object"},
        seed=7,
        max_tokens=64,
    )
    return GenerationRequest(**{**base, **kw})


def client(backend, **kw):
    return OpenAICompatibleClient(
        GemmaConfig(base_url=backend.url, model="gemma-test", api_key="k", **kw)
    )


def test_client_sends_seed_schema_and_auth_and_reports_real_usage(backend):
    resp = asyncio.run(client(backend).generate(req()))
    path, headers, body = backend.seen[0]
    assert path == "/v1/chat/completions"
    assert headers["Authorization"] == "Bearer k"
    assert body["model"] == "gemma-test" and body["seed"] == 7 and body["temperature"] == 0
    assert body["max_tokens"] == 64
    assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
    assert (resp.prompt_tokens, resp.completion_tokens, resp.total_tokens) == (50, 10, 60)
    assert resp.parsed["facts"][0]["value"] == "Paris"


def test_structured_output_can_be_disabled(backend):
    asyncio.run(client(backend, structured=False).generate(req()))
    assert "response_format" not in backend.seen[0][2]


def test_backend_errors_are_raised_not_hidden(backend):
    backend.status = 500
    with pytest.raises(ModelError, match="HTTP 500"):
        asyncio.run(client(backend, max_retries=0).generate(req()))
    backend.status, backend.usage = 200, False
    with pytest.raises(ModelError, match="token usage"):
        asyncio.run(client(backend).generate(req()))


def test_rate_limits_and_server_errors_are_retried(backend):
    backend.fail_first = [429, 503]
    resp = asyncio.run(client(backend, backoff_s=0).generate(req()))
    assert len(backend.seen) == 3 and resp.total_tokens == 60


def test_http_retries_are_charged_but_backoff_is_not_wall_budget(backend, tmp_path, monkeypatch):
    from runtime import gemma_client

    clock = [0.0]
    monkeypatch.setattr(gemma_client.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(
        gemma_client.time, "sleep", lambda delay: clock.__setitem__(0, clock[0] + delay)
    )
    write_snapshot(tmp_path)
    backend.fail_first = [429, 503]
    task = make_task()
    dag, runner = build_runner(GENOME_A, task, tmp_path, client(backend, backoff_s=20))
    result = asyncio.run(drive(dag, runner, task))
    assert result.failure is None
    assert result.budget_usage.retries == 2
    assert result.budget_usage.wall_time_s < 1
    assert result.metrics.model_calls == 4


def test_exhausted_http_retries_report_attempt_usage(backend, tmp_path):
    write_snapshot(tmp_path)
    backend.status = 503
    task = make_task()
    dag, runner = build_runner(
        GENOME_A, task, tmp_path, client(backend, max_retries=1, backoff_s=0)
    )
    result = asyncio.run(drive(dag, runner, task))
    assert result.failure.kind is FailureKind.MODEL_ERROR
    assert result.budget_usage.retries == 1
    assert result.metrics.model_calls == 2


def test_retries_are_bounded_and_client_errors_are_not_retried(backend):
    backend.status = 429
    with pytest.raises(ModelError, match="HTTP 429"):
        asyncio.run(client(backend, max_retries=2, backoff_s=0).generate(req()))
    assert len(backend.seen) == 3
    backend.seen.clear()
    backend.status = 400
    with pytest.raises(ModelError, match="HTTP 400"):
        asyncio.run(client(backend, backoff_s=0).generate(req()))
    assert len(backend.seen) == 1


def test_unreachable_backend_is_reported_as_unavailable():
    c = OpenAICompatibleClient(
        GemmaConfig(base_url="http://127.0.0.1:9/v1", model="m", max_retries=0)
    )
    with pytest.raises(ModelUnavailableError):
        asyncio.run(c.generate(req()))


def test_config_from_env_requires_a_real_backend_and_never_invents_one():
    with pytest.raises(ModelUnavailableError, match="GEMMA_BASE_URL"):
        GemmaConfig.from_env({})
    cfg = GemmaConfig.from_env(
        {"GEMMA_BASE_URL": "http://h/v1", "GEMMA_MODEL": "gemma", "GEMMA_MODEL_REVISION": "r1"}
    )
    assert cfg.api_key is None and cfg.structured and cfg.revision == "r1"
    a, b = OpenAICompatibleClient(cfg), OpenAICompatibleClient(cfg)
    assert a.model_hash == b.model_hash
    other = OpenAICompatibleClient(
        GemmaConfig(base_url="http://h/v1", model="gemma", revision="r2")
    )
    assert other.model_hash != a.model_hash
    assert callable(client_from_env)


def test_cache_key_is_deterministic_and_sensitive():
    k = req().cache_key("m1")
    assert k == req().cache_key("m1")
    assert k != req().cache_key("m2")
    assert k != req(seed=8).cache_key("m1")
    assert k != req(prompt_template_version="v2").cache_key("m1")


def test_genome_a_runs_through_the_real_client_and_a_local_backend(backend, tmp_path):
    write_snapshot(tmp_path)
    task = make_task()
    dag, runner = build_runner(GENOME_A, task, tmp_path, client(backend))
    result = asyncio.run(drive(dag, runner, task))
    assert result.failure is None and result.answer.values == {"capital": "Paris"}
    assert result.evidence[0].spans[0].page_id == "p1"
    assert result.budget_usage.tokens == 120 and result.metrics.model_calls == 2
    # every request carried the versioned template identity and the structured schema
    sent = [b for _, _, b in backend.seen]
    assert len(sent) == 2 and all("response_format" in b for b in sent)
    assert PAGES["p1"][:20] in sent[0]["messages"][0]["content"]
    assert PROMPT_TEMPLATE_VERSION  # versioned prompts are part of run identity


def test_backend_outage_fails_the_run_instead_of_faking_an_answer(backend, tmp_path):
    write_snapshot(tmp_path)
    backend.status = 500
    task = make_task()
    dag, runner = build_runner(GENOME_A, task, tmp_path, client(backend, backoff_s=0))
    result = asyncio.run(drive(dag, runner, task))
    assert result.failure.kind is FailureKind.MODEL_ERROR and result.answer is None
