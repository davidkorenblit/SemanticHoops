"""
ingestion/downloader.py

Wraps yt-dlp to download sport videos from YouTube (or any yt-dlp-supported
platform) into `data/raw/videos/`.

Design decisions:
  - Runs yt-dlp as a subprocess so the main process stays importable without
    yt-dlp being installed (useful in testing environments).
  - Limits format to ≤1080p to control disk usage.
  - Returns a list of downloaded file paths for downstream pipeline steps.
"""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List

from loguru import logger


class VideoDownloader:
    """Downloads videos via yt-dlp and returns local file paths."""

    DEFAULT_FORMAT = "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/mp4"

    def __init__(self, output_dir: str | Path = "data/raw/videos") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download(self, url: str) -> Path:
        """Download a single video URL and return the output file path."""
        logger.info(f"Downloading: {url}")
        output_template = str(self.output_dir / "%(id)s.%(ext)s")

        cmd = [
            "yt-dlp",
            "--format", self.DEFAULT_FORMAT,
            "--output", output_template,
            "--no-playlist",
            "--quiet",
            "--print", "after_move:filepath",
            url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        file_path = Path(result.stdout.strip())
        logger.success(f"Downloaded → {file_path}")
        return file_path

    def download_batch(self, urls: List[str]) -> List[Path]:
        """Download multiple URLs, skipping failures with a warning."""
        paths: List[Path] = []
        for url in urls:
            try:
                paths.append(self.download(url))
            except subprocess.CalledProcessError as exc:
                logger.warning(f"Failed to download {url}: {exc.stderr[:200]}")
        return paths
