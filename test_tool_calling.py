"""
Test script to verify tool calling integration in UnifiedLegalAgent.
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm.ollama_client import OllamaClient
from src.agents.unified_agent import UnifiedLegalAgent
from src.rag.chroma_store import ChromaVectorStore
from src.rag.rag_retriever import RAGRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_tool_calling():
    """Test tool calling functionality."""

    print("=" * 80)
    print("Testing Tool Calling Integration")
    print("=" * 80)

    # 1. Initialize LLM client
    print("\n1. Initializing Ollama client...")
    llm_client = OllamaClient(model="llama3.1:8b")

    if not llm_client.test_connection():
        print("❌ Ollama not running. Please start it with: ollama serve")
        return False
    print("✓ Ollama connected")

    # 2. Initialize RAG components (optional but recommended)
    print("\n2. Initializing RAG components...")
    try:
        vector_store = ChromaVectorStore(
            persist_directory="data/chroma_db",
            collection_name="legal_documents",
            embedding_model="all-mpnet-base-v2",
            device="cpu"
        )
        rag_retriever = RAGRetriever(vector_store=vector_store)
        print(f"✓ RAG components initialized ({vector_store.collection.count()} documents)")
    except Exception as e:
        print(f"⚠️  RAG initialization failed: {e}")
        print("   Continuing without RAG...")
        rag_retriever = None
        vector_store = None

    # 3. Initialize Unified Agent with tools enabled
    print("\n3. Initializing Unified Agent with tools enabled...")
    agent = UnifiedLegalAgent(
        llm_client=llm_client,
        rag_retriever=rag_retriever,
        enable_safety_filter=True,
        enable_tools=True
    )
    print("✓ Agent initialized")

    # 4. Test queries that should trigger different tools
    print("\n4. Testing tool calling with various queries...")
    print("=" * 80)

    test_cases = [
        {
            "query": "What is the definition of liability?",
            "expected_tool": "get_legal_definition",
            "complexity": "simple"
        },
        {
            "query": "Analyze GDPR Article 6 and compare it with other data protection regulations in the documents",
            "expected_tool": "check_statute_reference + search_legal_documents",
            "complexity": "complex"
        },
        {
            "query": "I need a comprehensive analysis of force majeure clauses: define the term, find examples in contracts, and explain the legal implications",
            "expected_tool": "get_legal_definition + search_legal_documents",
            "complexity": "complex"
        },
    ]

    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'=' * 80}")
        print(f"Test {i}/{len(test_cases)}")
        print(f"Query: {test_case['query']}")
        print(f"Expected Tool: {test_case['expected_tool']}")
        print(f"Expected Complexity: {test_case['complexity']}")
        print(f"{'=' * 80}\n")

        try:
            response = agent.query(test_case['query'])
            print(f"Response:\n{response}\n")
            print("✓ Query processed successfully")

        except Exception as e:
            print(f"❌ Error processing query: {e}")
            logger.exception("Detailed error:")

        print()

    # 5. Test direct tool execution
    print("\n5. Testing direct tool execution...")
    print("=" * 80)

    from src.tools.legal_tools import LegalTools

    tools = LegalTools(vector_store=vector_store if rag_retriever else None)

    # Test get_legal_definition
    print("\nTest: get_legal_definition('contract')")
    result = tools.get_legal_definition("contract")
    print(f"Result: {result}")

    # Test check_statute_reference
    print("\nTest: check_statute_reference('GDPR Article 6')")
    result = tools.check_statute_reference("GDPR Article 6")
    print(f"Result: {result}")

    # Test search_legal_documents (if RAG available)
    if vector_store:
        print("\nTest: search_legal_documents('data protection')")
        result = tools.search_legal_documents("data protection", top_k=2)
        print(f"Result: {result}")
    else:
        print("\n⚠️  Skipping search_legal_documents test (no vector store)")

    print("\n" + "=" * 80)
    print("Tool calling integration test complete!")
    print("=" * 80)

    return True


if __name__ == "__main__":
    try:
        success = test_tool_calling()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        logger.exception("Detailed error:")
        sys.exit(1)
