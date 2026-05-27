"""sandbox_query_expansion.py
=================================================
Sandbox Evaluation Script for Tactical Query Expansion (CuPL Paradigm)

Takes a tactical concept, retrieves or generates 20 fine-grained physical descriptions
(visual atoms) using the Gemini API, caches them locally, encodes them using CLIP,
and evaluates similarity against a test video frame.
"""

import os
import sys
import json
import argparse
import re
from pathlib import Path
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Add project root to sys.path so we can import from src
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.models.embeddings import CLIPWrapper

# Load environment variables from .env
load_dotenv(dotenv_path=PROJECT_ROOT / ".env")

def get_gemini_expanded_queries(concept: str, api_key: str) -> list:
    """
    Decomposes a tactical concept into exactly 20 visual atoms using Gemini.
    Attempts to use the google-generativeai SDK if available; otherwise falls back
    to direct REST API HTTP requests.
    """
    prompt = (
        f"Decompose the basketball tactical concept \"{concept}\" into exactly 20 visual atoms.\n"
        "A visual atom is a physical, non-technical description of what a camera sees in the video "
        "(e.g., \"a player in a white jersey standing still with bent knees to block a defender's path\", "
        "NOT technical terms like \"a player setting a screen\" or \"pick and roll\").\n"
        "Return the result STRICTLY as a JSON list of strings, with no other text, markdown formatting, or explanations."
    )

    # 1. Attempt using google-generativeai SDK if installed
    try:
        import google.generativeai as genai
        print("[Gemini] Attempting to generate atoms via Google Generative AI SDK...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        response = model.generate_content(
            prompt,
            generation_config={"response_mime_type": "application/json"}
        )
        
        text_content = response.text
        if text_content:
            atoms = parse_json_list(text_content)
            if atoms and len(atoms) >= 15:
                print(f"[Gemini] Successfully retrieved {len(atoms)} visual atoms via SDK.")
                return atoms
    except ImportError:
        print("[Gemini] google-generativeai package not installed. Falling back to REST API...")
    except Exception as e:
        print(f"[Gemini] SDK generation failed: {e}. Falling back to REST API...")

    # 2. REST API fallback via requests or urllib
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    data = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "responseMimeType": "application/json"
        }
    }

    try:
        import requests
        print("[Gemini] Sending HTTP POST request to Gemini REST API...")
        response = requests.post(url, headers=headers, params=params, json=data, timeout=30)
        response.raise_for_status()
        res_json = response.json()
        
        text_content = res_json['candidates'][0]['content']['parts'][0]['text']
        atoms = parse_json_list(text_content)
        if atoms:
            print(f"[Gemini] Successfully retrieved {len(atoms)} visual atoms via REST API.")
            return atoms
    except Exception as e:
        print(f"[Gemini] REST API via requests failed: {e}. Attempting standard urllib fallback...")
        
    # 3. Standard urllib fallback (built-in, no external dependencies)
    import urllib.request
    import urllib.error
    
    req_url = f"{url}?key={api_key}"
    req_data = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        req_url,
        data=req_data,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            res_body = response.read().decode("utf-8")
            res_json = json.loads(res_body)
            text_content = res_json['candidates'][0]['content']['parts'][0]['text']
            atoms = parse_json_list(text_content)
            if atoms:
                print(f"[Gemini] Successfully retrieved {len(atoms)} visual atoms via urllib.")
                return atoms
    except Exception as e:
        print(f"[Gemini] Standard urllib fallback failed: {e}")
        
    raise RuntimeError("Failed to retrieve expanded queries from Gemini API across all methods.")

def parse_json_list(text: str) -> list:
    """Parses a JSON list from LLM output, cleaning up markdown code blocks if necessary."""
    clean_text = text.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```(?:json)?\n", "", clean_text)
        clean_text = re.sub(r"\n```$", "", clean_text)
    
    try:
        parsed = json.loads(clean_text)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
    except Exception as e:
        print(f"[Warning] Failed to parse output as JSON list: {e}")
        # Naive fallback parsing of plain text list items
        items = []
        for line in text.splitlines():
            line = line.strip().lstrip("*-0123456789. ")
            if line:
                items.append(line)
        if items:
            return items
    return []

def main():
    parser = argparse.ArgumentParser(description="Sandbox test for CuPL Query Expansion & Vector Averaging")
    parser.add_argument(
        "--concept",
        type=str,
        default="pick_and_roll",
        help="Tactical concept to expand (e.g. pick_and_roll, zone_defense, help_defense)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="data/processed/vid1/vid1_frame_0000.jpg",
        help="Path to the test image/frame to evaluate similarity against"
    )
    parser.add_argument(
        "--cache_path",
        type=str,
        default="data/tactical_lexicon.json",
        help="Path to the local lexicon cache file"
    )
    parser.add_argument(
        "--search",
        type=bool,
        default=True,
        help="Whether to perform a real Qdrant search and comparison"
    )
    args = parser.parse_args()

    # 1. Validate Image Path
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"[Error] Test image frame not found: {image_path}")
        print("Please specify a valid image path via --image.")
        sys.exit(1)

    # Convert concept name to lowercase alphanumeric slug for cache key
    concept_key = args.concept.strip().lower().replace(" ", "_")

    # 2. Local Lexicon Caching Logic
    cache_file = Path(args.cache_path)
    lexicon_cache = {}
    
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                lexicon_cache = json.load(f)
            print(f"[Cache] Loaded existing lexicon cache from {cache_file}.")
        except Exception as e:
            print(f"[Cache Warning] Failed to load cache file: {e}. Starting fresh.")
            
    visual_atoms = []
    
    if concept_key in lexicon_cache:
        print(f"[Cache] Hit! Found '{concept_key}' in local cache.")
        visual_atoms = lexicon_cache[concept_key]
    else:
        print(f"[Cache] Miss! No cached descriptions found for '{concept_key}'.")
        # Retrieve GEMINI API KEY
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("[Error] GEMINI_API_KEY environment variable is not set.")
            print("Please set it in your environment or in the .env file.")
            sys.exit(1)
            
        print(f"[Gemini] Requesting CuPL decomposition for concept: '{args.concept}'...")
        try:
            visual_atoms = get_gemini_expanded_queries(args.concept, api_key)
            
            # Save new visual atoms to cache
            lexicon_cache[concept_key] = visual_atoms
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(lexicon_cache, f, indent=2)
            print(f"[Cache] Saved expanded descriptions to cache: {cache_file}")
            
        except Exception as e:
            print(f"[Error] Failed to get descriptions: {e}")
            sys.exit(1)

    # 3. Print Visual Atoms
    print("\n" + "=" * 60)
    print(f"VISUAL ATOMS FOR '{args.concept.upper()}' (CuPL Decomposition):")
    print("=" * 60)
    for idx, atom in enumerate(visual_atoms, 1):
        print(f"{idx:02d}. {atom}")
    print("=" * 60 + "\n")

    # 4. CLIP Encoding and Vector Averaging
    print("[CLIP] Initializing CLIP model wrapper...")
    clip_model = CLIPWrapper()
    
    print(f"[CLIP] Encoding {len(visual_atoms)} visual atoms...")
    atom_embeddings = []
    for atom in visual_atoms:
        emb = clip_model.get_text_embedding(atom)
        atom_embeddings.append(emb)
        
    atom_embeddings = np.array(atom_embeddings)  # Shape: (N, 512)
    print(f"[CLIP] Successfully encoded atoms into matrix of shape {atom_embeddings.shape}")
    
    # Compute the mean vector across all 20 visual atoms
    mean_vector = np.mean(atom_embeddings, axis=0)
    # Normalize the averaged vector to unit length
    mean_vector_norm = mean_vector / np.linalg.norm(mean_vector)
    print(f"[CLIP] Computed and normalized vector average. Norm: {np.linalg.norm(mean_vector_norm):.4f}")

    # 5. Encode test image
    print(f"[CLIP] Encoding test frame: {image_path}...")
    image_embedding = clip_model.get_image_embedding(image_path)
    if image_embedding is None:
        print("[Error] Failed to encode the test frame.")
        sys.exit(1)
        
    image_vector = np.array(image_embedding)  # Shape: (512,)
    print(f"[CLIP] Successfully encoded test frame. Shape: {image_vector.shape}")

    # 6. Calculate Cosine Similarity
    # Since both vectors are unit vectors, their dot product is exactly the cosine similarity
    cosine_similarity = float(np.dot(mean_vector_norm, image_vector))
    
    # Compare with the similarity of the raw concept string (unexpanded)
    print(f"[CLIP] Encoding raw concept query: '{args.concept}'...")
    raw_query_vector = np.array(clip_model.get_text_embedding(args.concept))
    raw_cosine_similarity = float(np.dot(raw_query_vector, image_vector))

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS:")
    print("=" * 60)
    print(f"Tactical Concept   : {args.concept}")
    print(f"Test Image Frame   : {image_path}")
    print("-" * 60)
    print(f"Cosine Similarity (Averaged Visual Atoms) : {cosine_similarity:.4f}")
    print(f"Cosine Similarity (Raw Query Concept)     : {raw_cosine_similarity:.4f}")
    print("-" * 60)
    improvement = cosine_similarity - raw_cosine_similarity
    print(f"Difference (CuPL vs. Raw Query)           : {improvement:+.4f}")
    print("=" * 60 + "\n")

    # 7. Qdrant Search Evaluation
    if args.search:
        print("[Qdrant] Connecting to Qdrant...")
        qdrant_host = os.getenv("QDRANT_HOST", "localhost")
        qdrant_port = int(os.getenv("QDRANT_PORT", 6333))
        
        try:
            client = QdrantClient(host=qdrant_host, port=qdrant_port, timeout=5)
            collection_name = "nba_frames"
            
            # Verify if the collection exists, otherwise fallback
            collections = [c.name for c in client.get_collections().collections]
            
            if collection_name not in collections:
                available = ", ".join(collections) if collections else "None"
                print(f"[Qdrant Warning] Collection '{collection_name}' not found. Available: {available}")
                if collections:
                    collection_name = collections[0]
                    print(f"[Qdrant] Falling back to search in collection: '{collection_name}'")
                else:
                    raise ValueError("No collections exist in the Qdrant instance.")
            
            print(f"[Qdrant] Performing vector search in '{collection_name}'...")
            
            # Search using averaged CuPL vector
            cupl_hits = client.search(
                collection_name=collection_name,
                query_vector=mean_vector_norm.tolist(),
                limit=5,
                with_payload=True
            )
            
            # Search using raw query vector
            raw_hits = client.search(
                collection_name=collection_name,
                query_vector=raw_query_vector.tolist(),
                limit=5,
                with_payload=True
            )
            
            print("\n" + "=" * 60)
            print("QDRANT VECTOR SEARCH COMPARISON (Top-3 Results):")
            print("=" * 60)
            
            print("A) SEARCH RESULTS USING AVERAGED VISUAL ATOMS (CuPL):")
            print("-" * 60)
            if not cupl_hits:
                print("No results returned.")
            for rank, hit in enumerate(cupl_hits[:3], 1):
                payload = hit.payload or {}
                video_id = payload.get("video_id", "N/A")
                timestamp = payload.get("timestamp", "N/A")
                image_path_val = payload.get("image_path", "N/A")
                print(f"Rank {rank} | Score: {hit.score:.4f} | Video: {video_id} | Time: {timestamp}s | Path: {image_path_val}")
                
            print("\nB) SEARCH RESULTS USING RAW QUERY ('" + args.concept + "'):")
            print("-" * 60)
            if not raw_hits:
                print("No results returned.")
            for rank, hit in enumerate(raw_hits[:3], 1):
                payload = hit.payload or {}
                video_id = payload.get("video_id", "N/A")
                timestamp = payload.get("timestamp", "N/A")
                image_path_val = payload.get("image_path", "N/A")
                print(f"Rank {rank} | Score: {hit.score:.4f} | Video: {video_id} | Time: {timestamp}s | Path: {image_path_val}")
                
            print("=" * 60 + "\n")
            
        except Exception as e:
            print(f"[Qdrant Error] Could not complete Qdrant search: {e}")
            print("Make sure your Qdrant container is running and populated.")

if __name__ == "__main__":
    main()
