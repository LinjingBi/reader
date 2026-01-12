"""
embed.py
Embedding wrapper for paper abstracts (or any text list).
Uses BAAI/bge-small-en-v1.5 by default (fast, strong general embedding model).
"""

from __future__ import annotations
from typing import List, TYPE_CHECKING
from sentence_transformers import SentenceTransformer
import numpy as np

if TYPE_CHECKING:
    from fetch import Paper


def build_embedding_text(p: Paper) -> str:
    """
    Build the text we embed.
    NOTE: keywords help, but can dominate if too many / too generic.
    Keep it short + structured.
    """
    kws = ", ".join(p.keywords[:10])  # cap to avoid keyword domination
    parts = [
        f"TITLE: {p.title}",
        f"SUMMARY: {p.summary}",
        f"KEYWORDS: {kws}" if kws else "KEYWORDS:"
    ]
    return "\n".join(parts)


class Embedder:
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Loads a sentence transformer model.
        BAAI/bge-small-en-v1.5 is fast and strong for general embeddings.
        normalize_embeddings=True makes cosine similarity straightforward.
        """
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        embeds a list of texts -> returns numpy array [num_docs, dim]
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine-friendly
        )
        return np.asarray(embeddings, dtype=np.float32)

if __name__ == "__main__":
    # mini usage test
    texts = [
        "This paper introduces a new method for robotic grasping.",
        "We study diffusion models for image generation."
    ]
    embedder = Embedder()
    embs = embedder.encode(texts)
    print(embs.shape)
