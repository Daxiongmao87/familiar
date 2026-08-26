"""Local embedding wrapper with lazy model loading."""

from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np


class Embedder:
    """Lazy wrapper over fastembed.TextEmbedding."""

    def __init__(self, model_id: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model_id = model_id
        self._model = None
        self._dim: Optional[int] = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self._model_id

    @property
    def dim(self) -> Optional[int]:
        return self._dim

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                from fastembed import TextEmbedding
            except Exception as exc:
                raise RuntimeError(
                    f"fastembed is required for Embedder; install it ({exc}). "
                    "If offline, pre-provision the dependency."
                ) from exc
            try:
                self._model = TextEmbedding(model_name=self._model_id)
            except Exception as exc:
                raise RuntimeError(
                    f"failed to load embedding model {self._model_id!r}: {exc}. "
                    "If running offline, pre-download the model into the fastembed cache "
                    "or point the cache directory at one already populated."
                ) from exc

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        self._ensure_model()
        assert self._model is not None
        gen = self._model.embed(texts)
        if hasattr(gen, "__iter__") and not isinstance(gen, np.ndarray):
            vecs = list(gen)
        else:
            vecs = list(gen)
        arr = np.asarray(vecs, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.ndim == 1 and arr.size > 0:
            arr = arr.reshape(1, -1)
        if self._dim is None and arr.shape[0] > 0:
            self._dim = int(arr.shape[1])
        return arr