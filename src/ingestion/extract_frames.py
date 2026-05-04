import os
import cv2
import time
from pathlib import Path
from loguru import logger

# ---------------------------------------------------------------------------
# Configuration & Logging Setup
# ---------------------------------------------------------------------------
RAW_VIDEO_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
PROCESSED_FRAME_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "logs"
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov"}

def _ensure_directory(path: Path) -> None:
    """Create *path* if it does not exist."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Critical: Failed to create directory {path}: {exc}")
        raise

# וידוא קיום תיקיית לוגים לפני הגדרת ה-Logger
_ensure_directory(LOG_DIR)

# הגדרת Loguru: גם הדפסה למסך וגם שמירה לקובץ JSON סדרתי
logger.add(
    LOG_DIR / "ingestion.json",
    serialize=True,     # הופך כל שורת לוג לאובייקט JSON
    rotation="10 MB",   # מונע מהקובץ לגדול מדי
    level="INFO"
)

def _list_video_files(directory: Path) -> list[Path]:
    """Return a sorted list of video file paths inside *directory*."""
    if not directory.is_dir():
        logger.error(f"Raw video directory does not exist: {directory}")
        return []
    return sorted(
        [p for p in directory.iterdir() if p.suffix.lower() in VIDEO_EXTENSIONS and p.is_file()],
        key=lambda p: p.name,
    )

def _extract_one_fps(video_path: Path, output_dir: Path) -> None:
    """
    Extract 1 FPS frames sequentially.
    Optimized for CPU by avoiding random-access seeking (O(N^2)).
    """
    video_id = video_path.stem
    _ensure_directory(output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Unable to open video file: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        logger.error(f"Invalid FPS ({fps}) for {video_path.name}")
        cap.release()
        return

    # אינדקס למעקב אחרי השניות שאנחנו רוצים לחלץ
    next_target_sec = 0
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # חישוב זמן נוכחי על בסיס אינדקס פריימים (יותר אמין מ-POS_MSEC בחלק מהפורמטים)
        current_sec = int(frame_count / fps)

        if current_sec >= next_target_sec:
            frame_filename = f"{video_id}_frame_{next_target_sec:04d}.jpg"
            frame_path = output_dir / frame_filename
            
            if cv2.imwrite(str(frame_path), frame):
                logger.debug(f"Saved: {frame_filename}")
            else:
                logger.error(f"Write failed: {frame_filename}")
            
            next_target_sec += 1

        frame_count += 1

    cap.release()

def extract_frames() -> None:
    """Main pipeline execution."""
    video_files = _list_video_files(RAW_VIDEO_DIR)
    if not video_files:
        logger.info("No video files found.")
        return

    for video_path in video_files:
        video_id = video_path.stem
        out_dir = PROCESSED_FRAME_DIR / video_id

        # Idempotency Check
        if out_dir.is_dir() and any(out_dir.iterdir()):
            logger.info(f"Skipping {video_path.name}: Frames already exist.")
            continue

        start_time = time.perf_counter()
        logger.info(f"Processing: {video_path.name}")
        
        try:
            _extract_one_fps(video_path, out_dir)
            elapsed = time.perf_counter() - start_time
            logger.info(f"Finished {video_path.name} in {elapsed:.2f}s")
        except Exception as e:
            logger.exception(f"Failed to process {video_path.name}: {e}")

if __name__ == "__main__":
    extract_frames()