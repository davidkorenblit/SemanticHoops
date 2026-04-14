"""
models/base.py

Abstract base class (Protocol) for all embedding backends.

Plug-and-play contract: any new backbone (VideoMAE, InternVideo, etc.) must
implement `embed_image` and `embed_text`. This lets the downstream pipeline
swap models without touching retrieval or API code.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """
    Interface that every embedding backend must satisfy.

    All embed methods return L2-normalised float32 numpy arrays of shape
    (embedding_dim,), ready to be upserted into Qdrant.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the output embedding vector."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable name / version string for the backbone."""
        ...

    @abstractmethod
    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """
        Embed a single BGR image (as returned by OpenCV).

        Args:
            image: np.ndarray of shape (H, W, 3), dtype uint8, BGR channel order.

        Returns:
            np.ndarray of shape (embedding_dim,), dtype float32, L2-normalised.
        """
        ...

    @abstractmethod
    def embed_text(self, text: str) -> np.ndarray:
        """
        Embed a natural-language query string.

        Args:
            text: A plain text search query (e.g. "basketball slam dunk").

        Returns:
            np.ndarray of shape (embedding_dim,), dtype float32, L2-normalised.
        """
        ...

    # ------------------------------------------------------------------
    # Shared utility (available to all subclasses)
    # ------------------------------------------------------------------

    @staticmethod
    def l2_normalize(vector: np.ndarray) -> np.ndarray:
        """L2-normalise a 1-D numpy array (in-place safe)."""
        norm = np.linalg.norm(vector)
        return vector / (norm + 1e-8)
