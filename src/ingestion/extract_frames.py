
"""src/ingestion/extract_frames.py
=================================================
Ingestion pipeline – frame extraction from raw video files.

* Reads all video files from ``data/raw/`` (currently MP4 files).
* For each video extracts **exactly one frame per second** (1 FPS) using OpenCV.
* Saves frames as JPEG images under ``data/processed/<video_id>/``.
* Frame filenames embed the video identifier and the timestamp in seconds:
  ``<video_id>_frame_<SSSS>.jpg`` where ``SSSS`` is zero‑padded to 4 digits.
* Uses ``loguru`` to log processing latency for each video (start → finish).
* Provides a clear public ``extract_frames`` function and a convenient
  ``__main__`` entry‑point for manual execution.

The implementation is deliberately lightweight and Docker‑friendly – no heavy
memory allocations, minimal external dependencies, and explicit error handling.
"""

import os
import cv2
import time
from pathlib import Path
from loguru import logger

# ---------------------------------------------------------------------------
# Configuration constants – adjust if the project layout changes
# ---------------------------------------------------------------------------
RAW_VIDEO_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_FRAME_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}


def _ensure_directory(path: Path) -> None:
    """Create *path* if it does not exist.

    Parameters
    ----------
    path: Path
        Destination directory.
    """
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        logger.error(f"Failed to create directory {path}: {exc}")
        raise


def _list_video_files(directory: Path):
    """Return a sorted list of video file paths inside *directory*.

    Only files with extensions listed in ``VIDEO_EXTENSIONS`` are returned.
    """
    if not directory.is_dir():
        logger.error(f"Raw video directory does not exist: {directory}")
        return []
    return sorted(
        [p for p in directory.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file()],
        key=lambda p: p.name,
    )


def _extract_one_fps(video_path: Path, output_dir: Path) -> None:
    """Extract 1 FPS frames from *video_path* into *output_dir*.

    The function creates *output_dir* if necessary and writes JPEG files named
    ``<video_id>_frame_<SSSS>.jpg`` where ``SSSS`` is the timestamp in seconds.
    """
    video_id = video_path.stem  # filename without extension
    _ensure_directory(output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Unable to open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = int(total_frames / fps) if fps > 0 else 0
    logger.debug(
        f"Processing {video_path.name}: fps={fps:.2f}, total_frames={total_frames}, duration≈{duration_sec}s"
    )

    # Iterate over each whole second of the video
    for sec in range(duration_sec + 1):  # include last second if exact
        # Seek to the desired timestamp (milliseconds)
        cap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"Failed to read frame at {sec}s from {video_path.name}")
            continue
        # Construct filename with zero‑padded seconds (4 digits → up to 9999 s)
        frame_filename = f"{video_id}_frame_{sec:04d}.jpg"
        frame_path = output_dir / frame_filename
        # Encode as JPEG; ``cv2.imwrite`` returns a bool indicating success
        if not cv2.imwrite(str(frame_path), frame):
            logger.error(f"Failed to write frame image: {frame_path}")
        else:
            logger.debug(f"Saved frame: {frame_path.name}")

    cap.release()


def extract_frames() -> None:
    """Public entry point – iterates over all raw videos and extracts frames.

    Logs the latency (seconds) taken for each video using ``loguru``.
    """
    video_files = _list_video_files(RAW_VIDEO_DIR)
    if not video_files:
        logger.info("No video files found to process.")
        return

    for video_path in video_files:
        video_id = video_path.stem
        out_dir = PROCESSED_FRAME_DIR / video_id
        start_time = time.time()
        logger.info(f"Starting extraction for {video_path.name} into {out_dir}")
        _extract_one_fps(video_path, out_dir)
        elapsed = time.time() - start_time
        logger.info(f"Finished {video_path.name} – latency: {elapsed:.2f}s")


if __name__ == "__main__":
    # When executed as a script, run the extraction pipeline.
    extract_frames()

