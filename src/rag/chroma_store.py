"""
ChromaDB vector store for local RAG implementation.
Optimized for local LLM usage with proper configuration.
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import chromadb
from chromadb.config import Settings

from .embeddings import LocalEmbeddings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """
    ChromaDB-based vector store for local document retrieval.
    Uses local embeddings (sentence-transformers) - no API calls.
    """

    def __init__(
        self,
        persist_directory: str = "data/chroma_db",
        collection_name: str = "legal_documents",
        embedding_model: str = "all-mpnet-base-v2",
        device: str = "cpu",
        distance_metric: str = "cosine",
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory to persist the database
            collection_name: Name of the collection
            embedding_model: Sentence-transformer model name
            device: Device for embeddings ("cpu" or "cuda")
            distance_metric: Distance metric ("cosine", "l2", "ip")
        """
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.distance_metric = distance_metric

        # Create persist directory
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        # Initialize embedding model
        logger.info(f"Initializing embeddings: {embedding_model} on {device}")
        self.embedder = LocalEmbeddings(model_name=embedding_model, device=device)

        # Initialize ChromaDB client
        logger.info(f"Initializing ChromaDB at {persist_directory}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(anonymized_telemetry=False),
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Legal documents for RAG",
                "embedding_model": embedding_model,
                "distance_metric": distance_metric,
            },
        )

        doc_count = self.collection.count()
        logger.info(
            f"ChromaDB initialized. Collection '{collection_name}' has {doc_count} documents"
        )

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of document texts
            metadatas: Optional list of metadata dicts (one per document)
            ids: Optional list of document IDs (auto-generated if None)
        """
        if not documents:
            logger.warning("No documents to add")
            return

        logger.info(f"Adding {len(documents)} documents to ChromaDB")

        # Generate IDs if not provided
        if ids is None:
            existing_count = self.collection.count()
            ids = [f"doc_{existing_count + i}" for i in range(len(documents))]

        # Generate embeddings
        logger.info("Generating embeddings for documents")
        embeddings = self.embedder.embed_documents(documents)

        # Prepare metadatas
        if metadatas is None:
            metadatas = [{} for _ in documents]

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(documents), batch_size):
            batch_end = min(i + batch_size, len(documents))
            logger.info(f"Adding batch {i//batch_size + 1}: documents {i} to {batch_end}")

            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=documents[i:batch_end],
                metadatas=metadatas[i:batch_end],
            )

        logger.info(f"Successfully added {len(documents)} documents")

    def search(
        self,
        query: str,
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.

        Args:
            query: Search query text
            top_k: Number of results to return
            where: Metadata filter (e.g., {"source": "contract.pdf"})
            where_document: Document content filter

        Returns:
            List of results with content, metadata, distance, and similarity scores
        """
        # Generate query embedding
        query_embedding = self.embedder.embed_query(query)

        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            where_document=where_document,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted_results = []
        if results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                distance = (
                    results["distances"][0][i] if results["distances"] else None
                )

                # Convert distance to similarity score
                # For cosine distance: similarity = 1 - distance
                # For L2 distance: approximate with 1 / (1 + distance)
                if distance is not None:
                    if self.distance_metric == "cosine":
                        similarity = 1 - distance
                    else:
                        similarity = 1 / (1 + distance)
                else:
                    similarity = None

                formatted_results.append(
                    {
                        "content": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i]
                        if results["metadatas"]
                        else {},
                        "distance": distance,
                        "similarity": similarity,
                        "id": results["ids"][0][i] if results["ids"] else None,
                    }
                )

        return formatted_results

    def search_with_threshold(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search with similarity threshold filtering.

        Args:
            query: Search query text
            top_k: Number of results to retrieve (before filtering)
            similarity_threshold: Minimum similarity score (0-1)
            where: Metadata filter

        Returns:
            Filtered list of results with similarity >= threshold
        """
        results = self.search(query, top_k=top_k, where=where)

        # Filter by threshold
        filtered_results = [
            r for r in results if r["similarity"] and r["similarity"] >= similarity_threshold
        ]

        logger.info(
            f"Retrieved {len(results)} documents, {len(filtered_results)} passed threshold {similarity_threshold}"
        )

        return filtered_results

    def delete_documents(self, ids: List[str]) -> None:
        """
        Delete documents by ID.

        Args:
            ids: List of document IDs to delete
        """
        self.collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} documents")

    def delete_collection(self) -> None:
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection_name)
        logger.info(f"Deleted collection: {self.collection_name}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.

        Returns:
            Dictionary with statistics
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.dimension,
            "distance_metric": self.distance_metric,
            "device": self.embedder.device,
        }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return f"ChromaVectorStore(collection='{stats['collection_name']}', docs={stats['document_count']}, model='{stats['embedding_model']}')"


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize vector store
    store = ChromaVectorStore(
        persist_directory="data/chroma_db_test",
        collection_name="test_legal_docs",
        embedding_model="all-mpnet-base-v2",
        device="cpu",
    )

    # Sample legal documents
    sample_docs = [
        "A contract is a legally binding agreement between two or more parties that creates mutual obligations enforceable by law.",
        "Copyright is a type of intellectual property that protects original works of authorship including literary, dramatic, musical, and artistic works.",
        "A non-disclosure agreement (NDA) is a legal contract between parties that outlines confidential material that the parties wish to share but restrict access to by third parties.",
        "A trademark is a recognizable sign, design, or expression that identifies products or services from a particular source.",
        "Liability refers to a legal responsibility or obligation that arises from actions or agreements.",
    ]

    sample_metadatas = [
        {"source": "contracts_101.pdf", "type": "definition"},
        {"source": "ip_law_guide.pdf", "type": "definition"},
        {"source": "business_agreements.pdf", "type": "definition"},
        {"source": "ip_law_guide.pdf", "type": "definition"},
        {"source": "legal_terms.pdf", "type": "definition"},
    ]

    # Add documents
    print("\n=== Adding Documents ===")
    store.add_documents(documents=sample_docs, metadatas=sample_metadatas)

    # Print stats
    print(f"\n=== Vector Store Stats ===")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    # Test search
    print("\n=== Testing Search ===")
    query = "What is an NDA?"
    results = store.search(query, top_k=3)

    print(f"\nQuery: {query}")
    print(f"Found {len(results)} results:\n")

    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"  Similarity: {result['similarity']:.4f}")
        print(f"  Source: {result['metadata'].get('source')}")
        print(f"  Content: {result['content'][:100]}...")
        print()

    # Test search with threshold
    print("\n=== Testing Search with Threshold ===")
    filtered_results = store.search_with_threshold(
        query="contract definition", top_k=5, similarity_threshold=0.3
    )

    print(f"Query: contract definition")
    print(f"Found {len(filtered_results)} results above threshold:\n")

    for i, result in enumerate(filtered_results, 1):
        print(f"Result {i}:")
        print(f"  Similarity: {result['similarity']:.4f}")
        print(f"  Content: {result['content'][:80]}...")
        print()

    # Cleanup (optional)
    # store.delete_collection()
