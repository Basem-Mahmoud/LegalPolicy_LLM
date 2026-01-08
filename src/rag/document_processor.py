"""
Document processor for legal documents.
Handles loading, chunking, and preprocessing of various document formats.
"""

from typing import List, Dict, Any
from pathlib import Path
import logging

# Document loaders
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Document:
    """Simple document class to store text and metadata."""

    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.content = content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(content_length={len(self.content)}, metadata={self.metadata})"


class DocumentProcessor:
    """
    Process legal documents for RAG system.
    Supports PDF, TXT, and DOCX formats.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        documents_path: str = "data/documents",
    ):
        """
        Initialize document processor.

        Args:
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            documents_path: Path to documents directory
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.documents_path = Path(documents_path)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.supported_extensions = {
            ".pdf": self._load_pdf,
            ".txt": self._load_txt,
            ".docx": self._load_docx,
        }

    def load_documents(self) -> List[Document]:
        """
        Load all documents from the documents directory.

        Returns:
            List of Document objects
        """
        all_documents = []

        if not self.documents_path.exists():
            logger.warning(f"Documents path {self.documents_path} does not exist")
            return all_documents

        for file_path in self.documents_path.rglob("*"):
            if file_path.is_file() and file_path.suffix in self.supported_extensions:
                try:
                    logger.info(f"Loading document: {file_path.name}")
                    docs = self._load_document(file_path)
                    all_documents.extend(docs)
                    logger.info(f"Loaded {len(docs)} chunks from {file_path.name}")
                except Exception as e:
                    logger.error(f"Error loading {file_path.name}: {str(e)}")

        logger.info(f"Total documents loaded: {len(all_documents)}")
        return all_documents

    def _load_document(self, file_path: Path) -> List[Document]:
        """
        Load a single document and split into chunks.

        Args:
            file_path: Path to the document

        Returns:
            List of Document chunks
        """
        extension = file_path.suffix
        loader_func = self.supported_extensions.get(extension)

        if not loader_func:
            raise ValueError(f"Unsupported file format: {extension}")

        # Load the document
        raw_docs = loader_func(str(file_path))

        # Split into chunks
        chunks = self._split_documents(raw_docs, file_path)

        return chunks

    def _load_pdf(self, file_path: str) -> List[Any]:
        """Load PDF document."""
        loader = PyPDFLoader(file_path)
        return loader.load()

    def _load_txt(self, file_path: str) -> List[Any]:
        """Load text document."""
        loader = TextLoader(file_path)
        return loader.load()

    def _load_docx(self, file_path: str) -> List[Any]:
        """Load DOCX document."""
        loader = Docx2txtLoader(file_path)
        return loader.load()

    def _split_documents(self, raw_docs: List[Any], file_path: Path) -> List[Document]:
        """
        Split documents into chunks.

        Args:
            raw_docs: Raw documents from loaders
            file_path: Original file path for metadata

        Returns:
            List of Document chunks
        """
        chunks = []
        for i, doc in enumerate(raw_docs):
            # Split the document content
            text_chunks = self.text_splitter.split_text(doc.page_content)

            # Create Document objects with metadata
            for j, chunk_text in enumerate(text_chunks):
                metadata = {
                    "source": file_path.name,
                    "file_path": str(file_path),
                    "page": i,
                    "chunk": j,
                    "total_chunks": len(text_chunks),
                }
                # Add any existing metadata from the loader
                if hasattr(doc, "metadata"):
                    metadata.update(doc.metadata)

                chunks.append(Document(content=chunk_text, metadata=metadata))

        return chunks

    def process_text(self, text: str, source: str = "custom") -> List[Document]:
        """
        Process raw text into chunks.

        Args:
            text: Raw text to process
            source: Source identifier

        Returns:
            List of Document chunks
        """
        text_chunks = self.text_splitter.split_text(text)

        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            metadata = {"source": source, "chunk": i, "total_chunks": len(text_chunks)}
            chunks.append(Document(content=chunk_text, metadata=metadata))

        return chunks

    def preprocess_legal_text(self, text: str) -> str:
        """
        Preprocess legal text for better chunking and retrieval.

        Args:
            text: Raw legal text

        Returns:
            Preprocessed text
        """
        # Remove excessive whitespace
        text = " ".join(text.split())

        # Normalize section markers
        # (Add more preprocessing as needed for legal documents)

        return text


if __name__ == "__main__":
    # Example usage
    processor = DocumentProcessor()
    documents = processor.load_documents()
    print(f"Loaded {len(documents)} document chunks")

    if documents:
        print("\nFirst document chunk:")
        print(documents[0])
        print(f"\nContent preview: {documents[0].content[:200]}...")
