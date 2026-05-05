"""src/models/embeddings.py
=================================================
Embedding Layer – CLIP Wrapper for multimodal features.

* Provides a modular wrapper for OpenAI's CLIP model.
* Includes methods to extract normalized embeddings for both images and text.
* Automatically detects and utilizes GPU if available, falling back to CPU.
"""

import torch
import clip
from PIL import Image
from loguru import logger
from pathlib import Path
from typing import List, Optional
# Ensure logs directory exists for embeddings
LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger.add(LOG_DIR / "embeddings.json", serialize=True, level="INFO")

class CLIPWrapper:
    def __init__(self, model_name: str = "ViT-B/32", device: Optional[str] = None):
        """
        Initialize the CLIP model.
        """
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Loading CLIP model '{model_name}' on {self.device}...")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()  # Set to inference mode
        logger.info("Model loaded successfully.")

    def get_image_embedding(self, image_path: Path) -> Optional[List[float]]:
        """Extract normalized embedding vector for a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to open image {image_path}: {e}")
            return None
        # Preprocess and add batch dimension
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy().tolist()[0]

    def get_batch_images_embeddings(self, image_paths: List[Path], batch_size: int = 32) -> List[dict]:
        """Extract embeddings for a list of images in batches.

        Handles corrupted files gracefully, logs progress and latency,
        and clears CUDA cache after each batch to keep memory usage low.
        """
        import time
        embeddings: List[dict] = []
        total = len(image_paths)
        if total == 0:
            return embeddings
        start_total = time.perf_counter()
        for batch_start in range(0, total, batch_size):
            batch_paths = image_paths[batch_start:batch_start + batch_size]
            batch_images = []
            valid_paths = []
            for p in batch_paths:
                try:
                    img = Image.open(p).convert("RGB")
                    batch_images.append(self.preprocess(img))
                    valid_paths.append(p)
                except Exception as e:
                    logger.error(f"Failed to open image {p}: {e}")
            if not batch_images:
                continue
            batch_tensor = torch.stack(batch_images).to(self.device)
            batch_start_time = time.perf_counter()
            with torch.no_grad():
                batch_features = self.model.encode_image(batch_tensor)
                batch_features /= batch_features.norm(dim=-1, keepdim=True)
            batch_embeddings = batch_features.cpu().numpy().tolist()
            
            for path, emb in zip(valid_paths, batch_embeddings):
                filename = path.stem
                if "_frame_" in filename:
                    video_id, frame_str = filename.rsplit("_frame_", 1)
                    try:
                        timestamp = int(frame_str)
                    except ValueError:
                        timestamp = 0
                else:
                    video_id = filename
                    timestamp = 0
                    
                embeddings.append({
                    "embedding": emb,
                    "file_path": str(path),
                    "video_id": video_id,
                    "timestamp": timestamp
                })
                
            batch_elapsed = time.perf_counter() - batch_start_time
            logger.info(
                f"Processed batch {batch_start // batch_size + 1}/{(total + batch_size - 1) // batch_size} "
                f"({len(valid_paths)} images) in {batch_elapsed:.2f}s"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        total_elapsed = time.perf_counter() - start_total
        logger.info(f"Total batch embedding time: {total_elapsed:.2f}s for {total} images")
        return embeddings

    def get_text_embedding(self, text: str) -> List[float]:
        """Extract normalized embedding vector for a text query."""
        text_input = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy().tolist()[0]
