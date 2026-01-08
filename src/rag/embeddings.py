"""
Local embeddings using sentence-transformers.
Provides a clean interface for generating embeddings without API calls.
"""

import logging
from typing import List, Optional
from sentence_transformers import SentenceTransformer
import torch

logger = logging.getLogger(__name__)


class LocalEmbeddings:
    """
    Local embedding model using sentence-transformers.
    No API calls required - runs 100% locally.
    """

    def __init__(
        self,
        model_name: str = "all-mpnet-base-v2",
        device: str = "cpu",
        normalize: bool = True,
        batch_size: int = 32,
    ):
        """
        Initialize local embedding model.

        Args:
            model_name: Sentence-transformer model name
                - "all-mpnet-base-v2" (768 dim, best quality, recommended)
                - "all-MiniLM-L6-v2" (384 dim, faster, lighter)
                - "BAAI/bge-large-en-v1.5" (1024 dim, best for retrieval)
            device: Device to run on ("cpu", "cuda", "mps")
            normalize: Whether to normalize embeddings (recommended for cosine similarity)
            batch_size: Batch size for encoding
        """
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size

        # Auto-detect device if cuda specified but not available
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"

        # Load model
        logger.info(f"Loading embedding model: {model_name} on {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)
        self.dimension = self.model.get_sentence_embedding_dimension()

        logger.info(
            f"Embedding model loaded. Dimension: {self.dimension}, Device: {self.device}"
        )

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query text.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector as list of floats
        """
        embedding = self.model.encode(
            text, convert_to_numpy=True, normalize_embeddings=self.normalize
        )
        return embedding.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of document texts to embed

        Returns:
            List of embedding vectors
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=len(texts) > 100,  # Show progress for large batches
        )

        return embeddings.tolist()

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding dimension
        """
        return self.dimension

    def get_model_info(self) -> dict:
        """
        Get information about the embedding model.

        Returns:
            Dictionary with model information
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "device": self.device,
            "normalize": self.normalize,
            "batch_size": self.batch_size,
        }

    def __repr__(self) -> str:
        return f"LocalEmbeddings(model='{self.model_name}', dim={self.dimension}, device='{self.device}')"


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize embeddings
    embedder = LocalEmbeddings(model_name="all-mpnet-base-v2", device="cpu")

    # Test single query embedding
    query = "What is a non-disclosure agreement?"
    query_embedding = embedder.embed_query(query)
    print(f"\nQuery: {query}")
    print(f"Embedding dimension: {len(query_embedding)}")
    print(f"First 5 values: {query_embedding[:5]}")

    # Test batch document embeddings
    documents = [
        "A contract is a legally binding agreement between two or more parties.",
        "Copyright protects original works of authorship.",
        "A trademark is a recognizable sign, design, or expression.",
    ]

    doc_embeddings = embedder.embed_documents(documents)
    print(f"\n{len(documents)} documents embedded")
    print(f"Each embedding has {len(doc_embeddings[0])} dimensions")

    # Calculate similarity between query and documents
    import numpy as np

    query_vec = np.array(query_embedding)
    for i, doc in enumerate(documents):
        doc_vec = np.array(doc_embeddings[i])
        similarity = np.dot(query_vec, doc_vec)  # Cosine similarity (normalized)
        print(f"\nDocument {i+1} similarity: {similarity:.4f}")
        print(f"Text: {doc[:60]}...")

    # Model info
    print(f"\nModel info: {embedder.get_model_info()}")
