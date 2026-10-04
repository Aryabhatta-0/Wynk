"""External-baseline comparison harness (experiments/oss_baselines): offline tests only.

No framework (smolagents / CrewAI / LlamaIndex / LangGraph) and no model backend is needed:
runners are exercised with stdlib fake runner scripts and the proxy with a local fake upstream.
"""

from __future__ import annotations

import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from benchmarks import heldout
from benchmarks.heldout import HELDOUT_DIR, HELDOUT_SNAPSHOTS
from benchmarks.loader import benchmark_hash, load_splits, load_task_specs
from benchmarks.snapshot_store import SnapshotStore
from core.evidence import FieldEvidence
from core.genome import Genome
from core.payloads import Answer
from core.results import BudgetUsage, ExecutionResult, FailureInfo, FailureKind, Verdict
from evaluation.evidence import SnapshotEvidenceVerifier
from evaluation.gate import DeterministicEvaluator
from experiments.oss_baselines import baselines as B
from experiments.oss_baselines import report as R
from experiments.oss_baselines.proxy import CallRecord, MeteringProxy, summarize_calls

GOLDEN_HELDOUT_HASH = "f88b7fc4649b49fd581f6ae0f1e95cd0691209b144301576a800b270442f296d"
STORE = SnapshotStore(HELDOUT_SNAPSHOTS)
SPECS = load_task_specs(HELDOUT_DIR)


def pages(snapshot_id: str):
    return {p.page_id: p for p in STORE.pages(snapshot_id).pages}


# --- held-out split -----------------------------------------------------------------------------


def test_heldout_hash_is_frozen_and_disjoint_from_search_data():
    assert benchmark_hash(HELDOUT_DIR, STORE) == GOLDEN_HELDOUT_HASH
    main = load_task_specs()
    assert not set(SPECS) & set(main)
    assert set(load_splits(HELDOUT_DIR)) == {"test"}
    assert sorted(load_splits(HELDOUT_DIR)["test"]) == sorted(SPECS)
    # new entities: no held-out question names a company from the search benchmark
    main_companies = {c["name"] for c in heldout.TEST_COMPANIES} & {
        s.runtime.question.split(" fact sheet")[0].removeprefix("Per the ") for s in main.values()
    }
    assert main_companies == set()


def test_heldout_regenerates_byte_identically(tmp_path):
    heldout.build(tmp_path)
    assert benchmark_hash(tmp_path, SnapshotStore(tmp_path / "snapshots")) == GOLDEN_HELDOUT_HASH


def test_frozen_wynk_matches_the_selected_aco_incumbents_and_heldout():
    frozen = json.loads((B.HERE / "frozen_wynk.json").read_text(encoding="utf-8"))
    assert frozen["heldout_test"]["benchmark_hash"] == GOLDEN_HELDOUT_HASH
    for c in frozen["classes"].values():
        assert Genome.from_stages(c["genome_stages"]).genome_hash == c["genome_hash"]
        assert not set(c["train_tasks"] + c["validation_tasks"]) & set(SPECS)


# --- inputs never carry ground truth ------------------------------------------------------------


class _FakeProxy:
    def base_url(self, run_id):
        return f"http://127.0.0.1:1/r/{run_id}/v1"

    def calls(self, run_id):
        return []


def test_runner_input_is_runtime_view_only():
    spec = SPECS["TB-003"]
    bl = B.SubprocessBaseline(
        B.SYSTEMS["crewai"], _FakeProxy(), B.ModelSettings("m"), pages, "h", Path(".")
    )
    inp = bl.build_input(spec.runtime_view(), 0, "http://x")
    text = json.dumps(inp)
    assert "ground_truth" not in text and "matchers" not in text
    assert inp["task"] == spec.runtime_view().model_dump(mode="json")
    assert inp["model"]["api_key"] == "via-metering-proxy"
    for f in spec.runtime.answer_schema.fields:
        assert f"- {f.name} ({f.type.value}, required)" in inp["task_prompt"]


# --- output normalisation -----------------------------------------------------------------------

FIELDS = {"hq", "founded"}
GOOD = {"answer": {"hq": "Eindhoven", "founded": 1978}, "evidence": {}}


@pytest.mark.parametrize(
    "final",
    [
        GOOD,
        json.dumps(GOOD),
        "```json\n" + json.dumps(GOOD) + "\n```",
        "Here is the result:\n" + json.dumps(GOOD) + "\nThanks!",
        str(GOOD),  # Python-literal dict (e.g. str() of a CodeAgent final_answer)
        json.dumps({"hq": "Eindhoven", "founded": 1978}),  # bare fields
    ],
)
def test_parse_final_accepts_common_shapes(final):
    assert B.parse_final(final, FIELDS)["answer"] == {"hq": "Eindhoven", "founded": 1978}


@pytest.mark.parametrize("final", [None, "", "I could not find it.", "{not json", 42, []])
def test_parse_final_rejects_non_answers(final):
    assert B.parse_final(final, FIELDS) is None


def test_locate_citation_exact_whitespace_and_ellipsis():
    p = pages("TB-001")
    content = p["items"].content
    rec = content[content.index("{", 40) : content.index("}", 40) + 1]  # one record, multi-line
    spans, how = B.locate_citation(rec, p, "items")
    assert how == "exact" and len(spans) == 1
    flat = " ".join(rec.split())  # re-flowed onto one line
    spans, how = B.locate_citation(flat, p, "items")
    assert how == "ws_insensitive" and content[spans[0].char_start : spans[0].char_end] == rec
    first, second = flat[:40], flat[-40:]
    spans, how = B.locate_citation(f"{first} ... {second}", p, None)
    assert how == "ws_insensitive" and len(spans) == 2
    assert all(SnapshotEvidenceVerifier(STORE).is_valid(s, "TB-001") for s in spans)


def test_locate_citation_rejects_altered_or_trivial_quotes():
    p = pages("TA-001")
    assert B.locate_citation("Veldkamp Instruments was founded in 1979", p, None) == ([], None)
    assert B.locate_citation("} ... {", p, None) == ([], None)  # fragments too short
    assert B.locate_citation("   ", p, None) == ([], None)


def test_normalize_answer_builds_verifiable_spans_and_drops_unknown_fields():
    task = SPECS["TA-001"].runtime_view()
    parsed = {
        "answer": {"hq": "Eindhoven", "founded": 1978, "extra": 1},
        "evidence": {
            "hq": [{"page_id": "overview", "quote": "headquartered in Eindhoven"}],
            "founded": {"page_id": "overview", "quote": "founded in 1999"},  # not on the page
        },
    }
    answer, quotes = B.normalize_answer(parsed, task, pages("TA-001"))
    assert answer.values == {"hq": "Eindhoven", "founded": 1978}
    assert [fe.field for fe in answer.evidence] == ["hq"]
    assert quotes["founded"][0]["matched"] is None


# --- status + evaluation ------------------------------------------------------------------------


def _execution(task_id, values, cited=True, usage=None, failure=None):
    spec = SPECS[task_id]
    p = pages(spec.runtime.snapshot_id)
    ev = ()
    if cited and values is not None:
        page = next(iter(p.values()))
        ev = tuple(FieldEvidence(field=f, spans=(page.span(0, 5),)) for f in values)
    bl = B.SubprocessBaseline(
        B.SYSTEMS["crewai"], _FakeProxy(), B.ModelSettings("m"), pages, "h", Path(".")
    )
    return ExecutionResult(
        key=bl._key(spec.runtime_view(), 0),
        answer=None if values is None else Answer(values=values, evidence=ev),
        budget_usage=usage or BudgetUsage(tokens=100, wall_time_s=1.0, tool_calls=2),
        failure=failure,
    )


def test_classify_covers_every_status():
    task = SPECS["TA-001"].runtime_view()
    p = pages("TA-001")
    ok = {"hq": "Eindhoven", "founded": 1978}
    assert B.classify(_execution("TA-001", ok), task, p) is B.RunStatus.SUCCESS
    assert B.classify(_execution("TA-001", ok), task, p, timed_out=True) is B.RunStatus.TIMEOUT
    over = BudgetUsage(tokens=task.caps.tokens + 1, wall_time_s=1.0)
    assert B.classify(_execution("TA-001", ok, usage=over), task, p) is B.RunStatus.BUDGET_EXCEEDED
    err = FailureInfo(kind=FailureKind.EXECUTOR_ERROR, message="boom")
    ex = _execution("TA-001", None, failure=err)
    assert B.classify(ex, task, p) is B.RunStatus.RUNTIME_ERROR
    assert B.classify(_execution("TA-001", None), task, p) is B.RunStatus.INVALID_OUTPUT
    bad_type = {"hq": "Eindhoven", "founded": "1978"}
    assert B.classify(_execution("TA-001", bad_type), task, p) is B.RunStatus.INVALID_OUTPUT
    no_ev = _execution("TA-001", ok, cited=False)
    assert B.classify(no_ev, task, p) is B.RunStatus.MISSING_EVIDENCE


def test_official_evaluation_applies_caps_and_diagnostic_lifts_them():
    ev = DeterministicEvaluator(SnapshotEvidenceVerifier(STORE))
    spec = SPECS["TA-001"]
    ok = {"hq": "Eindhoven", "founded": 1978}
    over = BudgetUsage(tokens=spec.caps.tokens + 1, wall_time_s=1.0)
    official, diag = B.evaluate(ev, spec, _execution("TA-001", ok, usage=over))
    assert official.verdict is Verdict.INFEASIBLE and official.fitness == -1.0
    assert diag.verdict is Verdict.PASS
    official, _ = B.evaluate(ev, spec, _execution("TA-001", ok))
    assert official.verdict is Verdict.PASS and official.fitness > 1.0


# --- subprocess adapter: failures stay visible --------------------------------------------------

FAKE_RUNNER = """
import json, sys, time
inp = json.load(open(sys.argv[1], encoding="utf-8"))
mode = {mode!r}
if mode == "sleep":
    time.sleep(30)
if mode == "crash":
    sys.exit(3)
out = {{"ok": True, "framework": {{"name": "fake"}}, "agent_latency_s": 0.5,
       "tool_calls": [{{"tool": "read_page", "args": {{"page_id": "overview"}}}}],
       "final": json.dumps({{"answer": {{"hq": "Eindhoven", "founded": 1978}},
                             "evidence": {{"hq": "headquartered in Eindhoven",
                                           "founded": "founded in 1978"}}}})}}
if mode == "raise":
    out = {{"ok": False, "error": {{"type": "ValueError", "message": "bad"}}, "tool_calls": []}}
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"))
"""


@pytest.mark.parametrize(
    ("mode", "status"),
    [
        ("ok", B.RunStatus.SUCCESS),
        ("raise", B.RunStatus.RUNTIME_ERROR),
        ("crash", B.RunStatus.RUNTIME_ERROR),
        ("sleep", B.RunStatus.TIMEOUT),
    ],
)
def test_subprocess_baseline_serializes_every_outcome(tmp_path, monkeypatch, mode, status):
    (tmp_path / "fake_runner.py").write_text(FAKE_RUNNER.format(mode=mode), encoding="utf-8")
    monkeypatch.setattr(B, "RUNNERS", tmp_path)
    spec = B.SystemSpec("fake", "Fake Starter", "external", "fake_runner.py", None)
    monkeypatch.setattr(B.SystemSpec, "python", lambda self: sys.executable)
    bl = B.SubprocessBaseline(
        spec, _FakeProxy(), B.ModelSettings("m"), pages, "h", tmp_path / "raw", timeout_s=5
    )
    r = bl.run(SPECS["TA-001"].runtime_view(), 0)
    assert r.status is status
    assert (tmp_path / "raw" / "fake.TA-001.r0" / "input.json").is_file()
    if mode == "ok":
        assert r.execution.answer.values == {"hq": "Eindhoven", "founded": 1978}
        assert r.execution.budget_usage.tool_calls == 1
        assert r.latency_s == 0.5
    if mode == "sleep":
        assert r.timed_out and r.execution.answer is None


# --- metering proxy -----------------------------------------------------------------------------


class _Upstream(BaseHTTPRequestHandler):
    seen: list[dict] = []

    def do_POST(self):  # noqa: N802
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _Upstream.seen.append({"auth": self.headers.get("Authorization"), "body": body})
        usage = {"prompt_tokens": 11, "completion_tokens": 4, "cost": "0.0001"}
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            chunks = [
                {"choices": [{"delta": {"content": "hel"}}]},
                {"choices": [{"delta": {"content": "lo"}}]},
                {"choices": [], "usage": usage},
            ]
            for c in chunks:
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        data = json.dumps({"choices": [{"message": {"content": "hi"}}], "usage": usage}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


@pytest.fixture
def upstream():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Upstream.seen = []
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_proxy_meters_plain_and_streamed_calls_and_holds_the_key(upstream):
    import urllib.request

    proxy = MeteringProxy(upstream, "REAL-KEY").start()
    try:
        url = proxy.base_url("run1") + "/chat/completions"
        for stream in (False, True):
            body = {
                "model": "g",
                "temperature": 0,
                "max_tokens": 7,
                "stream": stream,
                "messages": [{"role": "user", "content": "x"}],
                "tools": [{"type": "function", "function": {"name": "read_page"}}],
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(body).encode(),
                method="POST",
                headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                resp.read()
        calls = proxy.calls("run1")
    finally:
        proxy.stop()
    assert [c.stream for c in calls] == [False, True]
    assert all(c.prompt_tokens == 11 and c.completion_tokens == 4 for c in calls)
    assert all(c.temperature == 0 and c.max_tokens == 7 and c.tools == ["read_page"] for c in calls)
    assert calls[1].response_body["streamed_text"] == "hello"
    assert all(s["auth"] == "Bearer REAL-KEY" for s in _Upstream.seen)
    assert _Upstream.seen[1]["body"]["stream_options"] == {"include_usage": True}
    tel = summarize_calls(calls)
    assert tel["llm_calls"] == 2 and tel["total_tokens"] == 30
    assert tel["cost_usd"] == pytest.approx(0.0002)


def test_summarize_calls_never_reports_partial_totals():
    a = CallRecord("r", 0, "chat/completions", 0.0, prompt_tokens=5, completion_tokens=1)
    b = CallRecord("r", 1, "chat/completions", 0.0)  # backend sent no usage
    tel = summarize_calls([a, b])
    assert tel["llm_calls"] == 2 and tel["total_tokens"] is None and tel["cost_usd"] is None
    assert tel["known_total_tokens"] == 6


# --- report -------------------------------------------------------------------------------------


def _row(system, task, fitness, verdict="PASS", status="SUCCESS", tokens=100, cls="A"):
    return {
        "system": system,
        "label": system,
        "task_id": task,
        "task_class": cls,
        "run_index": 0,
        "fitness": fitness,
        "verdict": verdict,
        "status": status,
        "latency_s": 10.0,
        "quote_supports_value_fraction": 1.0,
        "uncapped": {
            "all_fields_matched": verdict == "PASS",
            "field_accuracy": 1.0,
            "evidence_valid_fraction": 1.0,
        },
        "telemetry": {"total_tokens": tokens, "cost_usd": 0.001, "llm_calls": 2, "tool_calls": 1},
    }


def test_report_picks_strongest_baseline_and_guards_relative_changes():
    rows = [
        _row("wynk", "T1", 1.05),
        _row("wynk", "T2", 1.05),
        _row("crewai", "T1", 1.02),
        _row("crewai", "T2", 0.7, "FAIL", "MISSING_EVIDENCE"),
        _row("smolagents", "T1", -1.0, "INFEASIBLE", "BUDGET_EXCEEDED", tokens=None),
        _row("smolagents", "T2", -1.0, "INFEASIBLE", "BUDGET_EXCEEDED"),
    ]
    s = R.summarize(rows)
    assert s["best_external_baseline"] == "crewai"
    smol = s["comparisons"]["all"]["smolagents"]
    assert smol["fitness"]["relative"] is None  # baseline fitness < 0: no ratio
    assert s["tables"]["all"]["smolagents"]["tokens_missing"] == 1
    assert s["tables"]["all"]["smolagents"]["constraint_pass_rate"] == 0.0
    crew = s["comparisons"]["all"]["crewai"]
    assert crew["fitness"]["relative"] == pytest.approx((1.05 - 0.86) / 0.86)
    paired = s["paired_fitness"]["crewai"]
    assert (paired["tasks_wynk_better"], paired["tasks_wynk_worse"]) == (2, 0)
    assert R.markdown(s, None)  # renders
