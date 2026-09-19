"""Unit tests for dmd.embedder — must never touch the network or a real model."""

from __future__ import annotations

import sys

import pytest


def test_embedder_name_property_returns_model_id():
    from dmd.embedder import Embedder

    e = Embedder("custom-model-id")
    assert e.name == "custom-model-id"

    default = Embedder()
    assert default.name == "BAAI/bge-small-en-v1.5"


def test_embedder_dim_is_none_before_first_embed():
    from dmd.embedder import Embedder

    e = Embedder("some-model")
    # No embed() call yet — the lazy loader hasn't fired.
    assert e.dim is None


def test_embedder_lazy_load_failure_raises_runtime_error_mentioning_offline(
    monkeypatch: pytest.MonkeyPatch,
):
    """If the fastembed import itself fails, embed() must raise RuntimeError
    and the message must mention the offline precondition so the caller
    knows to pre-provision the cache rather than retry blindly."""
    # Forcing sys.modules['fastembed'] to None makes any
    # `import fastembed` / `from fastembed import TextEmbedding` raise
    # ImportError immediately. The embedder wraps that as RuntimeError.
    monkeypatch.setitem(sys.modules, "fastembed", None)

    from dmd.embedder import Embedder

    e = Embedder("some-model")
    with pytest.raises(RuntimeError, match=r"(?i)offline"):
        e.embed(["hello world"])
