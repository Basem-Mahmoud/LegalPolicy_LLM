"""
RAG Retriever for local LLM implementation.
Combines ChromaDB vector store with context formatting for LLM consumption.
"""

import logging
from typing import List, Dict, Any, Optional

from .chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


class RAGRetriever:
    """
    High-level RAG retriever that combines document retrieval with context formatting.
    Optimized for local LLM usage with smart filtering and context management.
    """

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        max_context_length: int = 3000,
    ):
        """
        Initialize RAG retriever.

        Args:
            vector_store: ChromaVectorStore instance
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score (0-1)
            max_context_length: Maximum context length in tokens (approximate)
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.max_context_length = max_context_length

        logger.info(
            f"RAGRetriever initialized: top_k={top_k}, threshold={similarity_threshold}"
        )

    def retrieve(
        self, query: str, where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query with similarity filtering.

        Args:
            query: User query
            where: Optional metadata filter

        Returns:
            List of relevant documents with metadata and similarity scores
        """
        logger.info(f"Retrieving documents for query: {query[:50]}...")

        results = self.vector_store.search_with_threshold(
            query=query,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold,
            where=where,
        )

        logger.info(f"Retrieved {len(results)} documents above threshold")

        return results

    def format_context(
        self, retrieved_docs: List[Dict[str, Any]], include_metadata: bool = True
    ) -> str:
        """
        Format retrieved documents into context string for LLM.

        Args:
            retrieved_docs: List of retrieved documents
            include_metadata: Whether to include source metadata

        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant documents found in the knowledge base."

        context_parts = []

        for i, doc in enumerate(retrieved_docs, 1):
            content = doc["content"]
            similarity = doc.get("similarity", 0)

            # Create document header
            if include_metadata:
                source = doc["metadata"].get("source", "Unknown")
                doc_type = doc["metadata"].get("type", "document")
                header = f"[Document {i} - Source: {source}, Type: {doc_type}, Relevance: {similarity:.2%}]"
            else:
                header = f"[Document {i}]"

            context_parts.append(f"\n{header}\n{content}\n")

            # Check approximate token count (rough estimate: 1 token ≈ 4 chars)
            current_length = len("".join(context_parts))
            if current_length * 0.25 > self.max_context_length:
                logger.info(
                    f"Context length limit reached. Using only {i} out of {len(retrieved_docs)} documents."
                )
                break

        formatted_context = "\n".join(context_parts)
        return formatted_context

    def retrieve_and_format(
        self, query: str, include_metadata: bool = True
    ) -> tuple[List[Dict[str, Any]], str]:
        """
        Retrieve documents and format them as context in one call.

        Args:
            query: User query
            include_metadata: Whether to include metadata in context

        Returns:
            Tuple of (retrieved documents, formatted context string)
        """
        docs = self.retrieve(query)
        context = self.format_context(docs, include_metadata=include_metadata)
        return docs, context

    def retrieve_for_comparison(
        self, queries: List[str]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Retrieve documents for multiple queries (useful for query expansion).

        Args:
            queries: List of query variations

        Returns:
            Dictionary mapping each query to its retrieved documents
        """
        results = {}
        for query in queries:
            results[query] = self.retrieve(query)

        return results

    def get_retriever_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the retriever configuration.

        Returns:
            Dictionary with retriever statistics
        """
        vector_store_stats = self.vector_store.get_stats()

        return {
            "vector_store": vector_store_stats,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "max_context_length": self.max_context_length,
        }

    def update_config(
        self,
        top_k: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        max_context_length: Optional[int] = None,
    ) -> None:
        """
        Update retriever configuration dynamically.

        Args:
            top_k: New top_k value
            similarity_threshold: New similarity threshold
            max_context_length: New max context length
        """
        if top_k is not None:
            self.top_k = top_k
            logger.info(f"Updated top_k to {top_k}")

        if similarity_threshold is not None:
            self.similarity_threshold = similarity_threshold
            logger.info(f"Updated similarity_threshold to {similarity_threshold}")

        if max_context_length is not None:
            self.max_context_length = max_context_length
            logger.info(f"Updated max_context_length to {max_context_length}")

    def __repr__(self) -> str:
        return f"RAGRetriever(top_k={self.top_k}, threshold={self.similarity_threshold}, max_context={self.max_context_length})"


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize components
    from chroma_store import ChromaVectorStore

    print("=== Initializing RAG System ===")
    vector_store = ChromaVectorStore(
        persist_directory="data/chroma_db_test",
        collection_name="test_legal_docs",
        embedding_model="all-mpnet-base-v2",
    )

    retriever = RAGRetriever(
        vector_store=vector_store, top_k=3, similarity_threshold=0.5, max_context_length=2000
    )

    # Add some test documents if collection is empty
    if vector_store.collection.count() == 0:
        print("\n=== Adding Test Documents ===")
        sample_docs = [
            "A contract is a legally binding agreement between two or more parties.",
            "Copyright protects original works of authorship.",
            "A non-disclosure agreement (NDA) protects confidential information.",
            "Trademark identifies products or services from a particular source.",
            "Liability is a legal responsibility arising from actions or agreements.",
        ]

        sample_metadatas = [
            {"source": "contract_law.pdf", "type": "definition"},
            {"source": "ip_law.pdf", "type": "definition"},
            {"source": "business_agreements.pdf", "type": "definition"},
            {"source": "ip_law.pdf", "type": "definition"},
            {"source": "legal_terms.pdf", "type": "definition"},
        ]

        vector_store.add_documents(documents=sample_docs, metadatas=sample_metadatas)

    # Test retrieval
    print("\n=== Testing Retrieval ===")
    query = "What is a non-disclosure agreement?"
    docs, context = retriever.retrieve_and_format(query)

    print(f"\nQuery: {query}")
    print(f"Retrieved {len(docs)} documents\n")
    print("=" * 80)
    print("FORMATTED CONTEXT FOR LLM:")
    print("=" * 80)
    print(context)
    print("=" * 80)

    # Test with different threshold
    print("\n=== Testing with Higher Threshold ===")
    retriever.update_config(similarity_threshold=0.7)
    docs, context = retriever.retrieve_and_format("contract definition")

    print(f"\nQuery: contract definition")
    print(f"Retrieved {len(docs)} documents with threshold 0.7\n")

    for i, doc in enumerate(docs, 1):
        print(f"{i}. Similarity: {doc['similarity']:.4f} - {doc['content'][:60]}...")

    # Print stats
    print("\n=== Retriever Stats ===")
    stats = retriever.get_retriever_stats()
    print(f"Vector Store: {stats['vector_store']['document_count']} documents")
    print(f"Embedding Model: {stats['vector_store']['embedding_model']}")
    print(f"Top K: {stats['top_k']}")
    print(f"Threshold: {stats['similarity_threshold']}")
