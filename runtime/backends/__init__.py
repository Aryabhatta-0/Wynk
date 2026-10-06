"""Model backends: adapters behind the provider-neutral ``runtime.model_client.ModelClient``.

Only entry points that construct clients (CLIs, the API) import from here; the runtime,
executors and optimizers never do.
"""

from __future__ import annotations

from collections.abc import Callable

from core.models import ModelEntry, ModelRegistryError
from runtime.backends import openai_compatible
from runtime.model_client import RegisteredModelClient

ADAPTERS: dict[str, Callable[[ModelEntry, str | None], RegisteredModelClient]] = {
    openai_compatible.ADAPTER: openai_compatible.registered_client,
}


def client_for(entry: ModelEntry, api_key: str | None = None) -> RegisteredModelClient:
    """The registry-bound client for a pinned entry, via the adapter it declares."""
    build = ADAPTERS.get(entry.adapter)
    if build is None:
        raise ModelRegistryError(f"entry {entry.name!r}: no adapter {entry.adapter!r}")
    return build(entry, api_key)
