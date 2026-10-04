"""Deterministic mock JSON API served from the SAME records the snapshot pages were rendered from.

``GET /<endpoint>?field=value`` -> JSON text. Filters are exact string-equality on record
fields. The unfiltered response of an endpoint is byte-identical to the snapshot page of the
same name (Class B pages ARE the unfiltered API response), so evidence spans cited from
either access path verify against the same page content.
"""

from __future__ import annotations

import json
from typing import Any

from benchmarks.snapshot_store import SnapshotStore


class UnknownEndpoint(KeyError):
    pass


def render_response(endpoint: str, results: list[dict[str, Any]]) -> str:
    body = {"count": len(results), "endpoint": endpoint, "results": results}
    return json.dumps(body, sort_keys=True, indent=1, ensure_ascii=True) + "\n"


class MockAPI:
    def __init__(self, store: SnapshotStore | None = None) -> None:
        self.store = store or SnapshotStore()

    def get(self, snapshot_id: str, endpoint: str, **filters: Any) -> str:
        endpoints = self.store.records(snapshot_id)
        if endpoint not in endpoints:
            raise UnknownEndpoint(f"{snapshot_id}: no endpoint {endpoint!r}")
        rows = [
            r
            for r in endpoints[endpoint]
            if all(str(r.get(k)) == str(v) for k, v in filters.items())
        ]
        return render_response(endpoint, rows)
