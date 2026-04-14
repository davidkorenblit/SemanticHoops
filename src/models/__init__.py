"""
models/__init__.py

Public surface of the models layer:
  - CLIPWrapper     : plug-and-play CLIP embedding wrapper
  - BaseEmbedder    : abstract base class for new backbone integrations
"""
from .base import BaseEmbedder
from .clip_wrapper import CLIPWrapper

__all__ = ["BaseEmbedder", "CLIPWrapper"]
