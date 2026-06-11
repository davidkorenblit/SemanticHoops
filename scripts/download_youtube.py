"""scripts/download_youtube.py
=================================================
Utility script to download YouTube compilation videos for benchmarking.
Supports both single URL download and batch downloading from a JSON configuration.

Usage (Single Video):
    python scripts/download_youtube.py --url "https://www.youtube.com/watch?v=..." --name "dunk_compilation"

Usage (Batch Download):
    python scripts/download_youtube.py --batch configs/youtube_download_list.json
"""

import argparse
import json
from pathlib import Path
import yt_dlp

def download_video(url: str, name: str, output_dir: Path) -> None:
    """Download a single video from YouTube using yt-dlp."""
    output_template = str(output_dir / f"{name}.%(ext)s")

    ydl_opts = {
        'format': 'bestvideo[ext=mp4]/bestvideo',
        'outtmpl': output_template,
        'quiet': False,
        'no_warnings': True,
        'nocheckcertificate': True,  # Bypass SSL/proxy interception verification
    }

    print(f"\n[YouTube] Downloading: {name} from {url} ...")
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"[YouTube] Successfully downloaded to {output_dir}/{name}.mp4")
    except Exception as e:
        print(f"[ERROR] Failed to download {name}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", type=str, help="YouTube video URL (for single download)")
    parser.add_argument("--name", type=str, help="Filename (without extension) to save as (for single download)")
    parser.add_argument("--batch", type=str, help="Path to JSON configuration file for batch downloading")
    parser.add_argument("--output_dir", type=str, default="data/raw", help="Directory to save the video(s)")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        batch_path = Path(args.batch)
        if not batch_path.exists():
            print(f"[ERROR] Batch configuration file not found at: {batch_path}")
            return

        print(f"[YouTube] Loading batch download list from: {batch_path}")
        try:
            with open(batch_path, "r", encoding="utf-8") as f:
                download_list = json.load(f)
        except Exception as e:
            print(f"[ERROR] Failed to parse JSON config: {e}")
            return

        print(f"[YouTube] Found {len(download_list)} videos to download.")
        for idx, (name, url) in enumerate(download_list.items(), start=1):
            print(f"\n--- Processing Video {idx}/{len(download_list)} ---")
            download_video(url, name, out_dir)
            
        print("\n[YouTube] Batch download process complete.")

    elif args.url and args.name:
        download_video(args.url, args.name, out_dir)
    else:
        print("[ERROR] You must provide either --batch <json_file> OR both --url <url> and --name <name>.")

if __name__ == "__main__":
    main()
