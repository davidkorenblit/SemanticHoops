import sys
from pathlib import Path

# Add project root to sys.path so we can import from src
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.models.embeddings import CLIPWrapper

def main():
    # 1. Paths definition (searching for .jpg files inside the data directory)
    data_dir = PROJECT_ROOT / "data"
    
    # Take up to 5 .jpg files for testing from anywhere inside data/
    image_files = list(data_dir.rglob("*.jpg"))[:5]  
    
    if not image_files:
        print(f"No .jpg files found in {data_dir}.")
        print("Please ensure you have extracted frames before running this check.")
        return

    print(f"Found {len(image_files)} sample images. Initializing model...")

    # 2. Model Initialization
    embedder = CLIPWrapper()

    # 3. Execution
    print(f"Processing {len(image_files)} images...")
    results = embedder.get_batch_images_embeddings(image_files, batch_size=2)

    # 4. Results Validation
    for res in results:
        print(f"Video ID: {res['video_id']} | Timestamp: {res['timestamp']}")
        print(f"Image Path: {res['file_path']}")
        print(f"Embedding length: {len(res['embedding'])} (Expected 512)")
        print("-" * 40)

if __name__ == "__main__":
    main()
