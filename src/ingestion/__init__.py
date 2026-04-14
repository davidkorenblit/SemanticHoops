"""
ingestion/__init__.py

Public surface of the ingestion layer:
  - VideoDownloader : downloads videos via yt-dlp
  - SmartSampler    : extracts semantically rich frames via adaptive FPS
"""
from .downloader import VideoDownloader
from .sampler import SmartSampler

__all__ = ["VideoDownloader", "SmartSampler"]
