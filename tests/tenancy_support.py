"""Test helpers for the #32 multi-tenant API: a router with one dev workspace and an
all-scope key, whose ``handle`` sends that key. Existing (#18-#31) API tests run through it, so
every same-tenant flow is exercised authenticated, scoped and quota-checked."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, BinaryIO

from api.product import ApiResponse, build_api

DEV_EMAIL = "owner@example.test"
DEV_WORKSPACE = "ws-test"
DEV_KEY = "wynk_sk_00000000000000a1_" + "T" * 43  # deterministic: survives a restart


class DevAPI:
    def __init__(self, router: Any, workspace_id: str, key: str) -> None:
        self.router = router
        self.workspace_id = workspace_id
        self.key = key

    def handle(
        self, method: str, target: str, headers: Mapping[str, str], rfile: BinaryIO
    ) -> ApiResponse:
        h = dict(headers)
        if not any(k.lower() == "authorization" for k in h):
            h["Authorization"] = f"Bearer {self.key}"
        return self.router.handle(method, target, h, rfile)

    def start_workers(self) -> Any:
        return self.router.start_workers()

    @property
    def tenant_api(self) -> Any:
        return self.router.tenant(self.workspace_id).api

    def __getattr__(self, name: str) -> Any:  # service / jobs / promotions / ...
        return getattr(self.router.tenant(self.workspace_id).api, name)


def dev_api(
    data_dir: Path | str,
    *args: Any,
    workspace_id: str = DEV_WORKSPACE,
    key: str = DEV_KEY,
    **kwargs: Any,
) -> DevAPI:
    router = build_api(data_dir, *args, **kwargs)
    router.bootstrap(DEV_EMAIL, workspace_id=workspace_id, key_plaintext=key)
    return DevAPI(router, workspace_id, key)
