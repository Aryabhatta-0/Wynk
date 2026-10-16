"""Deterministic secrets / PII policy for uploaded datasets (#33). No model, no heuristics.

Every value of an upload (CSV fields; JSONL strings and integers, nested ones included) and the
display filename are matched against a fixed, versioned set of rules. Each rule is one bounded
regular expression (no nested unbounded repetition, so matching stays linear in the value),
optionally confirmed by a checksum (Luhn for payment cards, ISO 7064 mod-97 for IBANs, the
SSA area / group / serial rules for US SSNs). A match is a *finding* of one ``Category``; the
``SensitiveDataPolicy`` assigns every category one ``Action``:

    block   the upload is refused (``sensitive_data_blocked``, 422); nothing is stored
    redact  each matched span is replaced by ``[REDACTED:<category>]`` before anything is
            stored: the stored canonical bytes, the content hash, the dataset identity and
            every experiment see the redacted value (the change is recorded, never silent)
    allow   the value is kept verbatim; the finding is still recorded

The decision is recorded on the upload (``SensitiveDataReport``: policy version, and per
category + column the action, how many values matched and the first row) and in the audit
trail. Neither ever holds a matched value or any part of it.

Defaults: every credential category and the government / financial identifiers block (a
credential must never be stored or reach a model provider); contact data (email, phone) is
redacted. ``WYNK_SENSITIVE_POLICY="email=allow,phone=block"`` overrides per category; a
credential category can only block or redact (``allow`` refuses to start).

Logs: ``install_log_redaction`` scrubs every log record of the process with the same rules.

Rules are recall-bounded by construction: they find the listed formats, not "all PII". Names,
addresses, free-text health details and anything without a fixed format are NOT detected.
"""

from __future__ import annotations

import logging
import os
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, PositiveInt

from ingestion.export import is_formula_like

POLICY_VERSION = "sensitive-data/1"


class Kind(StrEnum):
    SECRET = "secret"
    PII = "pii"


class Category(StrEnum):
    PRIVATE_KEY = "private_key"
    AWS_ACCESS_KEY = "aws_access_key"
    GITHUB_TOKEN = "github_token"
    SLACK_TOKEN = "slack_token"
    STRIPE_SECRET_KEY = "stripe_secret_key"
    GOOGLE_API_KEY = "google_api_key"
    MODEL_API_KEY = "model_api_key"  # sk-... / sk-ant-... / sk-proj-...
    WYNK_API_KEY = "wynk_api_key"
    JWT = "jwt"
    PAYMENT_CARD = "payment_card"
    US_SSN = "us_ssn"
    IBAN = "iban"
    EMAIL = "email"
    PHONE = "phone"


class Action(StrEnum):
    ALLOW = "allow"
    REDACT = "redact"
    BLOCK = "block"


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = ord(ch) - 48
        if i % 2:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


def _card(match: str) -> bool:
    digits = re.sub(r"[ -]", "", match)
    if not 13 <= len(digits) <= 19 or not _luhn(digits):
        return False
    # issuer prefixes (IIN) of the major networks; an arbitrary Luhn-valid number is not a card
    two, four = int(digits[:2]), int(digits[:4])
    return (
        (digits[0] == "4" and len(digits) in (13, 16, 19))  # Visa
        or (51 <= two <= 55 and len(digits) == 16)  # Mastercard
        or (2221 <= four <= 2720 and len(digits) == 16)
        or (two in (34, 37) and len(digits) == 15)  # Amex
        or ((four == 6011 or two == 65 or 644 <= int(digits[:3]) <= 649) and len(digits) >= 16)
    )


def _ssn(match: str) -> bool:
    area, group, serial = match.split("-")
    return area not in ("000", "666") and area[0] != "9" and group != "00" and serial != "0000"


def _iban(match: str) -> bool:
    s = match.replace(" ", "").upper()
    if not 15 <= len(s) <= 34:
        return False
    moved = s[4:] + s[:4]
    return int("".join(str(int(c, 36)) for c in moved)) % 97 == 1


@dataclass(frozen=True)
class Rule:
    category: Category
    kind: Kind
    pattern: re.Pattern[str]
    confirm: Callable[[str], bool] | None = None


_B = r"(?<![A-Za-z0-9_])"  # left boundary that also treats _ as a word character
_E = r"(?![A-Za-z0-9_])"
RULES: tuple[Rule, ...] = (
    Rule(
        Category.PRIVATE_KEY,
        Kind.SECRET,
        re.compile(r"-----BEGIN (?:[A-Z0-9]{1,20} ){0,3}PRIVATE KEY-----"),
    ),
    Rule(Category.AWS_ACCESS_KEY, Kind.SECRET, re.compile(_B + r"(?:AKIA|ASIA)[0-9A-Z]{16}" + _E)),
    Rule(
        Category.GITHUB_TOKEN,
        Kind.SECRET,
        re.compile(_B + r"(?:gh[pousr]_[A-Za-z0-9]{36,255}|github_pat_[A-Za-z0-9_]{22,255})" + _E),
    ),
    Rule(Category.SLACK_TOKEN, Kind.SECRET, re.compile(_B + r"xox[abposr]-[A-Za-z0-9-]{10,250}")),
    Rule(
        Category.STRIPE_SECRET_KEY,
        Kind.SECRET,
        re.compile(_B + r"(?:sk|rk)_live_[A-Za-z0-9]{16,247}" + _E),
    ),
    Rule(Category.GOOGLE_API_KEY, Kind.SECRET, re.compile(_B + r"AIza[0-9A-Za-z_-]{35}" + _E)),
    Rule(
        Category.WYNK_API_KEY,
        Kind.SECRET,
        re.compile(_B + r"wynk_sk_[0-9a-f]{16}_[A-Za-z0-9_-]{43}"),
    ),
    Rule(
        Category.MODEL_API_KEY,
        Kind.SECRET,
        re.compile(r"(?<![A-Za-z0-9_-])sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,250}"),
    ),
    Rule(
        Category.JWT,
        Kind.SECRET,
        re.compile(
            _B + r"eyJ[A-Za-z0-9_-]{8,2048}\.eyJ[A-Za-z0-9_-]{8,8192}\.[A-Za-z0-9_-]{8,2048}"
        ),
    ),
    Rule(
        Category.PAYMENT_CARD,
        Kind.PII,
        re.compile(r"(?<![0-9])[0-9](?:[ -]?[0-9]){12,18}(?![0-9])"),
        _card,
    ),
    Rule(
        Category.US_SSN,
        Kind.PII,
        re.compile(r"(?<![0-9-])[0-9]{3}-[0-9]{2}-[0-9]{4}(?![0-9-])"),
        _ssn,
    ),
    Rule(
        Category.IBAN,
        Kind.PII,
        re.compile(_B + r"[A-Z]{2}[0-9]{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?" + _E),
        _iban,
    ),
    Rule(
        Category.EMAIL,
        Kind.PII,
        re.compile(
            r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@"
            r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
            r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?){0,8}\.[A-Za-z]{2,24}"
            r"(?![A-Za-z0-9-])"
        ),
    ),
    Rule(
        Category.PHONE,
        Kind.PII,
        # E.164 written with its leading +: country code then 7-13 more digits, single
        # separators allowed between digits. A bare digit run is never a phone number here.
        re.compile(r"(?<![\w+])\+[1-9](?:[ .-]?[0-9]){7,14}(?![0-9])"),
    ),
)
_RULE = {r.category: r for r in RULES}

DEFAULT_ACTIONS: dict[Category, Action] = {
    **{r.category: Action.BLOCK for r in RULES if r.kind is Kind.SECRET},
    Category.PAYMENT_CARD: Action.BLOCK,
    Category.US_SSN: Action.BLOCK,
    Category.IBAN: Action.BLOCK,
    Category.EMAIL: Action.REDACT,
    Category.PHONE: Action.REDACT,
}


def placeholder(category: Category) -> str:
    return f"[REDACTED:{category.value}]"


class SensitiveDataPolicy(BaseModel):
    """One action per category. Total: a category missing from ``actions`` is a config error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    actions: dict[Category, Action] = Field(default_factory=lambda: dict(DEFAULT_ACTIONS))

    def model_post_init(self, _: Any) -> None:
        missing = set(Category) - set(self.actions)
        if missing:
            raise ValueError(f"the policy has no action for {sorted(c.value for c in missing)}")
        secrets = {r.category for r in RULES if r.kind is Kind.SECRET}
        allowed = sorted(c.value for c in secrets if self.actions[c] is Action.ALLOW)
        if allowed:  # a credential is never stored: block or redact it
            raise ValueError(f"secrets cannot be allowed: {allowed}")

    def action(self, category: Category) -> Action:
        return self.actions[category]

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> SensitiveDataPolicy:
        """``WYNK_SENSITIVE_POLICY="email=allow,phone=block"``: per-category overrides of the
        defaults. An unknown category or action refuses to start (fail closed)."""
        env = os.environ if env is None else env
        raw = env.get("WYNK_SENSITIVE_POLICY", "").strip()
        actions = dict(DEFAULT_ACTIONS)
        for item in filter(None, (p.strip() for p in raw.split(","))):
            name, sep, value = item.partition("=")
            if not sep:
                raise ValueError(f"WYNK_SENSITIVE_POLICY entry {item!r} is not category=action")
            actions[Category(name.strip())] = Action(value.strip())
        return cls(actions=actions)


class Finding(BaseModel):
    """What matched where, and what was done. Never the value, a prefix or a hash of it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    category: Category
    kind: Kind
    action: Action
    column: str | None  # None: the display filename
    values: PositiveInt  # how many values matched at least once
    first_row: NonNegativeInt  # 1-based data row of the first match; 0 for the filename


class SensitiveDataReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    policy_version: str = POLICY_VERSION
    findings: tuple[Finding, ...] = ()
    redacted_values: NonNegativeInt = 0  # values rewritten in the stored bytes
    # CSV-injection prefixes (``ingestion.export``) detected after redaction, in any format
    formula_like_values: NonNegativeInt = 0
    # of those, rewritten in the stored bytes: every one for CSV, none for JSONL (#33)
    neutralized_values: NonNegativeInt = 0

    @property
    def blocked(self) -> tuple[Finding, ...]:
        return tuple(f for f in self.findings if f.action is Action.BLOCK)

    def columns_with_findings(self) -> set[str]:
        return {f.column for f in self.findings if f.column is not None}


# A value that contains none of these cannot match any rule (every rule needs one of them), so
# the full rule set only runs on candidates. Purely a speed-up: it never changes a result.
_TRIGGER = re.compile(
    r"@|[0-9][ .-]?[0-9][ .-]?[0-9]|[A-Z]{2}[0-9]{2}|-----BEGIN|gh[pousr]_|github_pat_|xox"
    r"|_live_|AIza|AKIA|ASIA|wynk_sk_|sk-|eyJ"
)
_MIN_MATCH = 6  # the shortest possible match: an email such as a@b.cd


def find(text: str) -> list[tuple[int, int, Category]]:
    """Every confirmed, non-overlapping match in ``text``: ``(start, end, category)``. Rules
    are applied in ``RULES`` order; a later rule never matches inside an earlier match."""
    if len(text) < _MIN_MATCH or _TRIGGER.search(text) is None:
        return []
    spans: list[tuple[int, int, Category]] = []
    for rule in RULES:
        for m in rule.pattern.finditer(text):
            if rule.confirm is not None and not rule.confirm(m.group()):
                continue
            s, e = m.span()
            if any(s < pe and ps < e for ps, pe, _ in spans):
                continue
            spans.append((s, e, rule.category))
    spans.sort()
    return spans


def redact_text(
    text: str, spans: list[tuple[int, int, Category]], policy: SensitiveDataPolicy
) -> str:
    out, last = [], 0
    for s, e, cat in spans:
        if policy.action(cat) is Action.REDACT:
            out.append(text[last:s])
            out.append(placeholder(cat))
            last = e
    out.append(text[last:])
    return "".join(out)


@dataclass
class _Tally:
    values: int = 0
    first_row: int = 0


@dataclass(frozen=True)
class Screened:
    report: SensitiveDataReport
    rows: tuple[dict[str, Any], ...] | None  # redacted rows; None when nothing was rewritten
    filename: str | None


def screen(
    names: list[str],
    rows: Iterable[Mapping[str, Any]],
    policy: SensitiveDataPolicy,
    filename: str | None = None,
) -> Screened:
    """Apply ``policy`` to every value. Pure and deterministic: the same rows and policy always
    give the same report and the same redacted rows."""
    tallies: dict[tuple[Category, str | None], _Tally] = {}
    redacted = 0
    formula_like = 0
    changed = False
    out_rows: list[dict[str, Any]] = []

    def note(cat: Category, column: str | None, row: int) -> None:
        t = tallies.setdefault((cat, column), _Tally())
        t.values += 1
        if t.values == 1:
            t.first_row = row

    def scan_str(value: str, column: str, row: int) -> str:
        nonlocal redacted, changed
        spans = find(value)
        if not spans:
            return value
        for cat in {c for _, _, c in spans}:
            note(cat, column, row)
        new = redact_text(value, spans, policy)
        if new != value:
            redacted += 1
            changed = True
        return new

    def walk(value: Any, column: str, row: int) -> Any:
        if isinstance(value, str):
            return scan_str(value, column, row)
        if isinstance(value, bool) or value is None or isinstance(value, float):
            return value
        if isinstance(value, int):
            text = str(value)
            out = scan_str(text, column, row)
            return value if out == text else out
        if isinstance(value, list):
            return [walk(v, column, row) for v in value]
        if isinstance(value, dict):
            return {k: walk(v, column, row) for k, v in value.items()}
        return value

    for i, row in enumerate(rows, start=1):
        new_row: dict[str, Any] = {}
        for name in names:
            new_row[name] = walk(row.get(name), name, i)
            value = new_row[name]  # counted as stored: after any redaction
            if isinstance(value, str) and is_formula_like(value):
                formula_like += 1
        out_rows.append(new_row)

    new_filename = filename
    if filename is not None:
        spans = find(filename)
        for cat in {c for _, _, c in spans}:
            note(cat, None, 0)
        new_filename = redact_text(filename, spans, policy) if spans else filename

    findings = tuple(
        Finding(
            category=cat,
            kind=_RULE[cat].kind,
            action=policy.action(cat),
            column=column,
            values=t.values,
            first_row=t.first_row,
        )
        for (cat, column), t in sorted(
            tallies.items(), key=lambda kv: (kv[1].first_row, kv[0][1] or "", kv[0][0].value)
        )
    )
    report = SensitiveDataReport(
        findings=findings, redacted_values=redacted, formula_like_values=formula_like
    )
    return Screened(report, tuple(out_rows) if changed else None, new_filename)


# -- logs ------------------------------------------------------------------------------------
_SCRUB = SensitiveDataPolicy(actions=dict.fromkeys(Category, Action.REDACT))


def scrub(text: str) -> str:
    """``text`` with every rule match replaced by its placeholder, whatever the policy."""
    return redact_text(text, find(text), _SCRUB)


_installed = False
_install_lock = threading.Lock()


def install_log_redaction() -> None:
    """Scrub every log record of the process (message, arguments and formatted traceback)
    with the same rules, as it is created - before any handler or formatter sees it. Defence
    in depth: the code never logs uploaded values, but an exception message might quote one.
    Idempotent."""
    global _installed
    with _install_lock:
        if _installed:
            return
        previous = logging.getLogRecordFactory()

        def factory(*args: Any, **kwargs: Any) -> logging.LogRecord:
            record = previous(*args, **kwargs)
            try:
                message = record.getMessage()
            except Exception:  # a malformed format call: leave it to the handler
                return record
            clean = scrub(message)
            if clean != message:
                record.msg, record.args = clean, None
            if record.exc_info and record.exc_text is None:
                record.exc_text = scrub(logging.Formatter().formatException(record.exc_info))
            return record

        logging.setLogRecordFactory(factory)
        _installed = True
