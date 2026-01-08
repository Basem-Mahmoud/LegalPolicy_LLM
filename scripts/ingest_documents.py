"""
Document Ingestion Script for Local LLM Setup.
Processes legal documents and ingests them into ChromaDB vector store.
"""

import argparse
import logging
from pathlib import Path
import yaml
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.document_processor import DocumentProcessor
from src.rag.chroma_store import ChromaVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config_local.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def ingest_documents(
    documents_dir: str, config: dict, recreate_collection: bool = False
) -> None:
    """
    Ingest documents into vector store.

    Args:
        documents_dir: Directory containing documents to ingest
        config: Configuration dictionary
        recreate_collection: Whether to delete and recreate the collection
    """
    logger.info("=" * 80)
    logger.info("Starting document ingestion process")
    logger.info("=" * 80)

    # Initialize document processor
    logger.info(f"Loading documents from: {documents_dir}")
    processor = DocumentProcessor(
        chunk_size=config["rag"]["chunk_size"],
        chunk_overlap=config["rag"]["chunk_overlap"],
        documents_path=documents_dir,
    )

    # Load and process documents
    documents = processor.load_documents()
    logger.info(f"Loaded {len(documents)} document chunks")

    if len(documents) == 0:
        logger.warning("No documents found to ingest!")
        logger.warning(f"Please add documents to {documents_dir}")
        logger.warning("Supported formats: .pdf, .txt, .docx")
        return

    # Initialize vector store
    logger.info("Initializing ChromaDB vector store...")
    vector_store = ChromaVectorStore(
        persist_directory=config["vector_store"]["persist_directory"],
        collection_name=config["vector_store"]["collection_name"],
        embedding_model=config["embeddings"]["model"],
        device=config["embeddings"]["device"],
        distance_metric=config["vector_store"]["distance_metric"],
    )

    # Recreate collection if requested
    if recreate_collection:
        logger.warning("Recreating collection (deleting existing data)...")
        try:
            vector_store.delete_collection()
            # Reinitialize
            vector_store = ChromaVectorStore(
                persist_directory=config["vector_store"]["persist_directory"],
                collection_name=config["vector_store"]["collection_name"],
                embedding_model=config["embeddings"]["model"],
                device=config["embeddings"]["device"],
                distance_metric=config["vector_store"]["distance_metric"],
            )
            logger.info("Collection recreated")
        except Exception as e:
            logger.error(f"Error recreating collection: {e}")

    # Extract texts and metadatas from Document objects
    texts = [doc.content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    # Add documents to vector store
    logger.info("Adding documents to vector store...")
    logger.info("This may take a few minutes depending on document count...")

    try:
        vector_store.add_documents(documents=texts, metadatas=metadatas)
        logger.info("✓ Documents successfully ingested!")
    except Exception as e:
        logger.error(f"Error ingesting documents: {e}")
        raise

    # Print statistics
    logger.info("\n" + "=" * 80)
    logger.info("INGESTION COMPLETE")
    logger.info("=" * 80)

    stats = vector_store.get_stats()
    logger.info(f"Collection: {stats['collection_name']}")
    logger.info(f"Total documents: {stats['document_count']}")
    logger.info(f"Embedding model: {stats['embedding_model']}")
    logger.info(f"Embedding dimension: {stats['embedding_dimension']}")
    logger.info(f"Device: {stats['device']}")
    logger.info(f"Distance metric: {stats['distance_metric']}")
    logger.info("=" * 80)

    # Test search
    logger.info("\nTesting search functionality...")
    test_query = "contract definition"
    results = vector_store.search(test_query, top_k=3)

    logger.info(f"Test query: '{test_query}'")
    logger.info(f"Found {len(results)} results:\n")

    for i, result in enumerate(results[:3], 1):
        logger.info(f"Result {i}:")
        logger.info(f"  Similarity: {result['similarity']:.4f}")
        logger.info(f"  Source: {result['metadata'].get('source', 'Unknown')}")
        logger.info(f"  Preview: {result['content'][:100]}...")
        logger.info("")

    logger.info("✓ Vector store is ready for use!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Ingest legal documents into ChromaDB vector store"
    )
    parser.add_argument(
        "--documents",
        type=str,
        default="data/documents",
        help="Directory containing documents to ingest (default: data/documents)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_local.yaml",
        help="Path to configuration file (default: config/config_local.yaml)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate the collection (WARNING: deletes existing data)",
    )

    args = parser.parse_args()

    # Verify documents directory exists
    docs_path = Path(args.documents)
    if not docs_path.exists():
        logger.error(f"Documents directory not found: {args.documents}")
        logger.error("Please create the directory and add documents.")
        sys.exit(1)

    # Verify config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    # Load configuration
    logger.info(f"Loading configuration from: {args.config}")
    config = load_config(args.config)

    # Run ingestion
    try:
        ingest_documents(
            documents_dir=args.documents,
            config=config,
            recreate_collection=args.recreate,
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
