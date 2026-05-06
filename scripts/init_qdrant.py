import os
import argparse
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "nba_frames"
VECTOR_SIZE = 512

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Delete the collection before creating it.")
    args = parser.parse_args()
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "qdrant"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )

    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if args.reset:
            client.delete_collection(COLLECTION_NAME)
            print(f"Collection '{COLLECTION_NAME}' deleted.")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists. Skipping.")
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    print(f"Collection '{COLLECTION_NAME}' created successfully.")

if __name__ == "__main__":
    main()
