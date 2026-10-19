"""The live-evidence-provider authority a durable job, its provenance and workflow memory bind.

#55 bound ONE string, ``evidence_provider``: the pinned JEV identity, or ``None``. A runtime can
now hold two live providers at once (JEV and websearch), so the authority is a SET of provider
identities, encoded in that same single string so every existing value keeps its exact bytes:

    no live provider       None                               (unchanged)
    one provider           that provider's identity           (a JEV-only value is unchanged)
    several providers      canonical JSON list, sorted        '["jev:...","websearch:..."]'

A provider identity never starts with ``[`` (JEV identities start with ``jev:``, websearch ones
with ``websearch:``), so the encoding is unambiguous. Equality of the encoded string is equality
of the set, so every place that pins the authority (job identity, warm-start pins, validity keys)
still compares it exactly: changing EITHER configured provider changes the authority and fails
closed. Where a candidate's own provider (``RunVersions.evidence_source``) is checked against
the authority, the check is membership (``authority_members``), never equality: a candidate
uses at most one provider, the one it actually gathered with.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any

from core.canonical import canonical_json

# Every websearch provider identity starts with this (``runtime.sources.WebSearchSource``).
WEBSEARCH_IDENTITY_PREFIX = "websearch:"


def is_websearch_identity(identity: str | None) -> bool:
    return identity is not None and identity.startswith(WEBSEARCH_IDENTITY_PREFIX)


def compose_authority(*identities: str | None) -> str | None:
    """The authority over ``identities`` (``None`` entries ignored)."""
    members = sorted({i for i in identities if i is not None})
    for i in members:
        if not i or i.startswith("["):
            raise ValueError(f"not a provider identity: {i!r}")
    if not members:
        return None
    if len(members) == 1:
        return members[0]
    return canonical_json(members)


def authority_members(authority: str | None) -> frozenset[str]:
    """The provider identities ``authority`` binds (empty for ``None``). Raises ``ValueError``
    for a value ``compose_authority`` would never produce."""
    if authority is None:
        return frozenset()
    if not authority.startswith("["):
        return frozenset({authority})
    try:
        members = json.loads(authority)
    except json.JSONDecodeError:
        raise ValueError("malformed evidence-provider authority") from None
    if (
        not isinstance(members, list)
        or len(members) < 2
        or not all(isinstance(m, str) for m in members)
        or compose_authority(*members) != authority
    ):
        raise ValueError("malformed evidence-provider authority")
    return frozenset(members)


def unauthorized(authority: str | None, used: Iterable[str | None]) -> tuple[str, ...]:
    """The providers in ``used`` (candidates' ``evidence_source``) that ``authority`` does not
    bind, sorted. ``None`` (a frozen fetch / api / inline run) is always authorized."""
    allowed = authority_members(authority)
    return tuple(sorted({u for u in used if u is not None and u not in allowed}))


def live_websearch(authority: str | None, run_versions: Iterable[Mapping[str, Any]]) -> bool:
    """True when ``authority`` binds a websearch provider or any of ``run_versions`` (serialized
    ``RunVersions``) gathered with one: live evidence that is not a reproducible snapshot."""
    used = [rv.get("evidence_source") for rv in run_versions]
    return any(is_websearch_identity(p) for p in (*authority_members(authority), *used))
