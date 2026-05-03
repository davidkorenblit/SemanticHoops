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
            # Preprocess and add batch dimension
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # Encode image and normalize (L2)
                image_features = self.model.encode_image(image_input)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                
            # Return as a standard python list of floats (ready for Vector DB)
            return image_features.cpu().numpy().tolist()[0]
        except Exception as e:
            logger.error(f"Failed to extract embedding for {image_path}: {e}")
            return None

    def get_text_embedding(self, text: str) -> List[float]:
        """Extract normalized embedding vector for a text query."""
        text_input = clip.tokenize([text]).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(text_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features.cpu().numpy().tolist()[0]
