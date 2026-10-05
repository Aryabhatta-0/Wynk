import asyncio

import pytest

from core.evidence import FieldEvidence
from core.payloads import Answer, Fact, Facts, Page, Pages
from core.results import FailureKind
from core.stages import VerifyMethod
from core.task_spec import AnswerField, AnswerSchema, FieldType
from runtime.executors.base import ExecutorInput
from runtime.executors.verify import VerifyExecutor, type_ok
from tests.conftest import make_task, verify
from tests.runtime_helpers import PAGES, QUOTE, make_ctx

PAGE = Page(page_id="p1", source_ref="x", content=PAGES["p1"])
START = PAGES["p1"].index(QUOTE)
GOOD = PAGE.span(START, START + len(QUOTE))


def run(payload, method, pages=(PAGE,), task=None):
    task = task or make_task()
    inp = ExecutorInput(stage_index=3, stage=verify(method), payload=payload, source_pages=pages)
    return asyncio.run(VerifyExecutor().run(inp, make_ctx(task)))


def facts(*fs):
    return Facts(facts=tuple(fs))


@pytest.mark.parametrize(
    ("value", "ftype", "ok"),
    [
        ("x", FieldType.STRING, True),
        (1, FieldType.STRING, False),
        (3, FieldType.INTEGER, True),
        (True, FieldType.INTEGER, False),
        (2.5, FieldType.NUMBER, True),
        (True, FieldType.NUMBER, False),
        (False, FieldType.BOOLEAN, True),
        ("2024-02-30", FieldType.DATE, False),
        ("2024-02-29", FieldType.DATE, True),
        (["a", "b"], FieldType.STRING_LIST, True),
        (["a", 1], FieldType.STRING_LIST, False),
    ],
)
def test_type_checks(value, ftype, ok):
    assert type_ok(value, ftype) is ok


def test_schema_check_accepts_valid_facts_and_answers():
    assert (
        run(facts(Fact(field="capital", value="Paris")), VerifyMethod.SCHEMA_CHECK).failure is None
    )
    out = run(Answer(values={"capital": "Paris"}), VerifyMethod.SCHEMA_CHECK)
    assert out.failure is None and out.payload == Answer(values={"capital": "Paris"})


def test_schema_check_rejects_missing_unknown_and_mistyped():
    missing = run(facts(), VerifyMethod.SCHEMA_CHECK)
    assert missing.failure.kind is FailureKind.SCHEMA_INVALID
    assert "missing required field 'capital'" in missing.failure.message
    assert (
        "unknown field"
        in run(Answer(values={"capital": "P", "x": 1}), VerifyMethod.SCHEMA_CHECK).failure.message
    )
    assert (
        "not a valid string"
        in run(Answer(values={"capital": 7}), VerifyMethod.SCHEMA_CHECK).failure.message
    )
    assert run(Answer(values={}), VerifyMethod.SCHEMA_CHECK).failure is not None


def test_optional_fields_may_be_absent():
    schema = AnswerSchema(
        fields=(
            AnswerField(name="capital", type=FieldType.STRING),
            AnswerField(name="river", type=FieldType.STRING, required=False),
        )
    )
    task = make_task(answer_schema=schema)
    assert (
        run(Answer(values={"capital": "Paris"}), VerifyMethod.SCHEMA_CHECK, task=task).failure
        is None
    )


def test_evidence_span_accepts_supported_facts_and_answers():
    fact = Fact(field="capital", value="Paris", spans=(GOOD,))
    assert run(facts(fact), VerifyMethod.EVIDENCE_SPAN).failure is None
    ans = Answer(
        values={"capital": "Paris"}, evidence=(FieldEvidence(field="capital", spans=(GOOD,)),)
    )
    assert run(ans, VerifyMethod.EVIDENCE_SPAN).failure is None


@pytest.mark.parametrize(
    ("payload", "needle"),
    [
        (facts(Fact(field="capital", value="Paris")), "no evidence"),
        (
            facts(
                Fact(
                    field="capital",
                    value="Paris",
                    spans=(GOOD.model_copy(update={"content_hash": "0" * 64}),),
                )
            ),
            "does not match the source page",
        ),
        (
            facts(Fact(field="capital", value="Paris", spans=(PAGE.span(0, 10_000),))),
            "does not match the source page",
        ),
        (facts(Fact(field="capital", value="Lyon", spans=(GOOD,))), "does not contain the value"),
        (Answer(values={"capital": "Paris"}), "no evidence"),
    ],
)
def test_evidence_span_rejects_unsupported_evidence(payload, needle):
    out = run(payload, VerifyMethod.EVIDENCE_SPAN)
    assert out.failure.kind is FailureKind.SCHEMA_INVALID and needle in out.failure.message


def test_evidence_span_needs_source_pages_and_self_consistency_is_not_in_the_mvp():
    fact = Fact(field="capital", value="Paris", spans=(GOOD,))
    assert (
        "no source pages" in run(facts(fact), VerifyMethod.EVIDENCE_SPAN, pages=()).failure.message
    )
    out = run(facts(fact), VerifyMethod.SELF_CONSISTENCY)
    assert out.failure.kind is FailureKind.EXECUTOR_ERROR
    assert run(Pages(), VerifyMethod.SCHEMA_CHECK).failure.kind is FailureKind.EXECUTOR_ERROR
