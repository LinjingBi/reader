"""
Embedding wrapper for paper abstracts (or any text list).
Uses BAAI/bge-small-en-v1.5 by default (fast, strong general embedding model).
"""

from __future__ import annotations
from typing import Sequence
from sentence_transformers import SentenceTransformer
import numpy as np

from algo_lib.typing import PaperLike



class Embedder:
    """
    Embedder for encoding papers into embeddings.
    """
    
    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        """
        Loads a sentence transformer model.
        BAAI/bge-small-en-v1.5 is fast and strong for general embeddings.
        normalize_embeddings=True makes cosine similarity straightforward.
        """
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
    
    def _build_embedding_texts(self,
            papers: Sequence[PaperLike],
            mode: str,
            top_n: int = 10,
        ) -> list[str]:
        """
        Build embedding texts from papers according to mode.
        
        Args:
            papers: Sequence of paper-like objects
            mode: Embedding mode
                - "A": title + summary
                - "B": title + summary + top_n keywords
                - "C": title + summary + all keywords
            top_n: Number of top keywords to use for mode "B"
        
        Returns:
            List of text strings ready for embedding
        
        Raises:
            ValueError: If mode is not "A", "B", or "C"
        """
        texts = []
        for p in papers:
            base = f"TITLE: {p.title}\nSUMMARY: {p.summary}".strip()

            if mode == "A":
                texts.append(base)
            elif mode == "B":
                kws = p.keywords[:top_n]
                texts.append(base + "\nKEYWORDS: " + ", ".join(kws))
            elif mode == "C":
                texts.append(base + "\nKEYWORDS: " + ", ".join(p.keywords))
            else:
                raise ValueError(f"Unknown mode: {mode}")
        
        return texts
    
    def encode_papers(
        self,
        papers: Sequence[PaperLike],
        *,
        mode: str,
        top_n: int = 10,
        batch_size: int = 50,
    ) -> np.ndarray:
        """
        Encode papers into embeddings.
        
        Args:
            papers: Sequence of paper-like objects to encode
            mode: Embedding mode
                - "A": title + summary
                - "B": title + summary + top_n keywords
                - "C": title + summary + all keywords
            top_n: Number of top keywords to use for mode "B"
            batch_size: Batch size for encoding
        
        Returns:
            Array of embeddings with shape (n_papers, embedding_dim)
        """
        texts = self._build_embedding_texts(papers, mode=mode, top_n=top_n)
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine-friendly
        )
        return np.asarray(embeddings, dtype=np.float32)
    
    def encode_texts(
        self,
        papers: Sequence[PaperLike],
        mode: str,
        top_n: int = 10,
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Alias for encode_papers for backward compatibility.
        """
        return self.encode_papers(
            papers,
            mode=mode,
            top_n=top_n,
            batch_size=batch_size,
        )

