import os
import shutil
import json
from pathlib import Path

def main():
    # התיקייה שאליה נרכז את 5 התמונות הטובות ביותר
    target_dir = Path("data/processed/top_results")
    target_dir.mkdir(parents=True, exist_ok=True)

    # נתיב לקובץ התוצאות שנוצר (נניח שזה הקובץ הכי חדש או קובץ ספציפי)
    # בהנחה שהקובץ האחרון נשמר תחת השם הזה:
    log_file_path = Path("data/logs/search_results.json")
    
    if not log_file_path.exists():
        print(f"Error: Could not find {log_file_path}")
        return

    with open(log_file_path, "r") as f:
        data = json.load(f)

    print(f"Extracting top matches for query: '{data.get('query')}'\n")

    for idx, match in enumerate(data.get("top_matches", [])):
        payload = match.get("payload", {})
        original_path = Path(payload.get("image_path"))
        
        # בניית שם חדש כדי שלא יידרסו תמונות עם אותו שם (למשל frame_0314 ממקורות שונים)
        new_filename = f"rank_{idx+1}_{payload.get('video_id')}_{original_path.name}"
        new_path = target_dir / new_filename

        if original_path.exists():
            shutil.copy2(original_path, new_path)
            print(f"Copied Rank {idx+1}: {original_path.name} -> {new_filename}")
        else:
            print(f"File not found: {original_path}")

    print(f"\nAll files copied to: {target_dir.absolute()}")

if __name__ == "__main__":
    main()