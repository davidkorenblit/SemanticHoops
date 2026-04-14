"""
models/clip_wrapper.py

Concrete implementation of BaseEmbedder backed by OpenAI CLIP.

Supports both `openai/clip` and `open_clip` (LAION-trained variants) via a
unified interface. The model name is resolved at init time from the environment.

Usage:
    embedder = CLIPWrapper(model_name="ViT-B/32", device="cuda")
    vec = embedder.embed_image(bgr_frame)            # shape: (512,)
    vec = embedder.embed_text("basketball dunk")     # shape: (512,)
"""
from __future__ import annotations

import numpy as np
import torch
import cv2
from PIL import Image

from .base import BaseEmbedder


# Lazy import so the module is importable even if CLIP isn't installed
# (useful in API containers that don't need the full ML stack).
def _load_clip(model_name: str, device: str):
    try:
        import clip  # openai-clip
        model, preprocess = clip.load(model_name, device=device)
        return model, preprocess, "openai"
    except ImportError:
        import open_clip  # open-clip-torch fallback
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        model = model.to(device)
        return model, preprocess, "open_clip"


class CLIPWrapper(BaseEmbedder):
    """
    CLIP embedding wrapper — plug-and-play backbone.

    Args:
        model_name: CLIP variant, e.g. "ViT-B/32", "ViT-L/14".
        device:     "cuda", "cpu", or "mps".
    """

    _DIM_MAP = {
        "ViT-B/32": 512,
        "ViT-B/16": 512,
        "ViT-L/14": 768,
        "ViT-L/14@336px": 768,
    }

    def __init__(
        self,
        model_name: str = "ViT-B/32",
        device: str | None = None,
    ) -> None:
        self._model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model, self._preprocess, self._backend = _load_clip(
            model_name, self.device
        )
        self._model.eval()

        self._dim = self._DIM_MAP.get(model_name, 512)

    # ------------------------------------------------------------------
    # BaseEmbedder contract
    # ------------------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"CLIP/{self._model_name} ({self._backend})"

    def embed_image(self, image: np.ndarray) -> np.ndarray:
        """Convert BGR numpy frame → CLIP embedding."""
        # OpenCV → PIL (RGB)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)

        tensor = self._preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self._model.encode_image(tensor)

        return self.l2_normalize(features.cpu().numpy().squeeze().astype(np.float32))

    def embed_text(self, text: str) -> np.ndarray:
        """Tokenise and encode a text query → CLIP embedding."""
        if self._backend == "openai":
            import clip
            tokens = clip.tokenize([text]).to(self.device)
        else:
            import open_clip
            tokens = open_clip.tokenize([text]).to(self.device)

        with torch.no_grad():
            features = self._model.encode_text(tokens)

        return self.l2_normalize(features.cpu().numpy().squeeze().astype(np.float32))
