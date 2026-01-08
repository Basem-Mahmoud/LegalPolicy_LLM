"""
Vector store for semantic search over legal documents.
Supports ChromaDB and FAISS backends.
"""

import os
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import yaml

from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from .document_processor import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VectorStore:
    """
    Vector store for document retrieval using semantic search.
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        db_path: str = "data/vector_db",
        collection_name: str = "legal_documents"
    ):
        """
        Initialize vector store.

        Args:
            embedding_model: Name of the sentence transformer model
            db_path: Path to persist the vector database
            collection_name: Name of the collection
        """
        self.embedding_model_name = embedding_model
        self.db_path = Path(db_path)
        self.collection_name = collection_name

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()

        # Initialize ChromaDB
        self.db_path.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Legal documents for RAG"}
        )

        logger.info(f"Vector store initialized with {self.collection.count()} documents")

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of Document objects to add
        """
        if not documents:
            logger.warning("No documents to add")
            return

        logger.info(f"Adding {len(documents)} documents to vector store")

        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        texts = []
        metadatas = []

        for i, doc in enumerate(documents):
            # Generate embedding
            embedding = self.embedding_model.encode(doc.content).tolist()

            # Create unique ID
            doc_id = f"{doc.metadata.get('source', 'unknown')}_{i}"

            ids.append(doc_id)
            embeddings.append(embedding)
            texts.append(doc.content)
            metadatas.append(doc.metadata)

        # Add to collection in batches
        batch_size = 100
        for i in range(0, len(ids), batch_size):
            batch_end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:batch_end],
                embeddings=embeddings[i:batch_end],
                documents=texts[i:batch_end],
                metadatas=metadatas[i:batch_end]
            )

        logger.info(f"Successfully added {len(documents)} documents")

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents using semantic similarity.

        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Optional metadata filter

        Returns:
            List of results with documents and metadata
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()

        # Search in collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )

        # Format results
        formatted_results = []
        if results['documents'] and results['documents'][0]:
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None,
                    'id': results['ids'][0][i] if results['ids'] else None
                })

        return formatted_results

    def similarity_search_with_threshold(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search with similarity threshold filtering.

        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)

        Returns:
            Filtered list of results
        """
        results = self.search(query, top_k)

        # Filter by similarity threshold
        # Note: ChromaDB returns distances, smaller is more similar
        # Convert to similarity score (1 - normalized_distance)
        filtered_results = []
        for result in results:
            # Approximate similarity from L2 distance
            # This is a rough approximation, adjust based on embedding model
            distance = result.get('distance', float('inf'))
            similarity = 1 / (1 + distance)  # Simple conversion

            if similarity >= similarity_threshold:
                result['similarity'] = similarity
                filtered_results.append(result)

        return filtered_results

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
            "embedding_dimension": self.embedding_dim,
            "embedding_model": self.embedding_model_name
        }


class RAGRetriever:
    """
    High-level RAG retriever that combines document retrieval with context formatting.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize RAG retriever.

        Args:
            vector_store: VectorStore instance
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity threshold
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: User query

        Returns:
            List of relevant documents with metadata
        """
        results = self.vector_store.similarity_search_with_threshold(
            query=query,
            top_k=self.top_k,
            similarity_threshold=self.similarity_threshold
        )
        return results

    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string for LLM.

        Args:
            retrieved_docs: List of retrieved documents

        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No relevant documents found."

        context_parts = ["Here are the relevant documents:\n"]

        for i, doc in enumerate(retrieved_docs, 1):
            source = doc['metadata'].get('source', 'Unknown')
            content = doc['content']

            context_parts.append(f"\n--- Document {i} (Source: {source}) ---")
            context_parts.append(content)

        return "\n".join(context_parts)

    def retrieve_and_format(self, query: str) -> tuple[List[Dict[str, Any]], str]:
        """
        Retrieve documents and format them as context.

        Args:
            query: User query

        Returns:
            Tuple of (retrieved documents, formatted context)
        """
        docs = self.retrieve(query)
        context = self.format_context(docs)
        return docs, context


if __name__ == "__main__":
    # Example usage
    from document_processor import DocumentProcessor

    # Process documents
    processor = DocumentProcessor()
    documents = processor.load_documents()

    # Create vector store and add documents
    vector_store = VectorStore()
    vector_store.add_documents(documents)

    # Search
    results = vector_store.search("What is a non-disclosure agreement?", top_k=3)
    print(f"Found {len(results)} results")

    for i, result in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Source: {result['metadata'].get('source')}")
        print(f"Content preview: {result['content'][:200]}...")
