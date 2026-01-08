"""
Test script to verify local LLM setup.
Tests all components: Ollama, embeddings, ChromaDB, and integrated system.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress

console = Console()
logging.basicConfig(level=logging.INFO)


def test_ollama():
    """Test Ollama connection and model availability using the official library."""
    console.print("\n[bold cyan]Testing Ollama...[/bold cyan]")

    target_model = "llama3.1:8b"

    try:
        import ollama

        # 1. Test connection by attempting to list models
        # The official client will raise a RequestError or ConnectionError if Ollama is down
        response = ollama.list()
        console.print("[green]✓ Ollama connection successful[/green]")

        # 2. List models and check for target
        # The response.models is now a list of Model objects
        models = [m.model for m in response.models]
        console.print(f"[green]✓ Available models: {len(models)}[/green]")

        if target_model not in models:
            console.print(f"[yellow]⚠ {target_model} not found[/yellow]")
            console.print(f"[yellow]Pull it with: ollama pull {target_model}[/yellow]")
            return False

        # 3. Test generation (Chat API)
        console.print(f"Testing text generation with {target_model}...")

        # Use ollama.chat for message-based structures
        chat_response = ollama.chat(
            model=target_model,
            messages=[{"role": "user", "content": "Say 'test successful' in one word"}],
        )

        # Access the content via response.message.content
        result_text = chat_response.message.content.strip()
        console.print(f"[green]✓ Generation works: {result_text}[/green]")

        return True

    except Exception as e:
        console.print(f"[red]✗ Ollama test failed: {e}[/red]")
        return False


def test_embeddings():
    """Test local embeddings."""
    console.print("\n[bold cyan]Testing Embeddings...[/bold cyan]")

    try:
        from src.rag.embeddings import LocalEmbeddings

        embedder = LocalEmbeddings(model_name="all-mpnet-base-v2", device="cpu")

        # Test single query
        query_embedding = embedder.embed_query("test query")
        console.print(
            f"[green]✓ Query embedding: {len(query_embedding)} dimensions[/green]"
        )

        # Test batch
        doc_embeddings = embedder.embed_documents(["doc 1", "doc 2", "doc 3"])
        console.print(
            f"[green]✓ Batch embedding: {len(doc_embeddings)} documents[/green]"
        )

        return True

    except Exception as e:
        console.print(f"[red]✗ Embeddings test failed: {e}[/red]")
        return False


def test_chroma():
    """Test ChromaDB vector store."""
    console.print("\n[bold cyan]Testing ChromaDB...[/bold cyan]")

    try:
        from src.rag.chroma_store import ChromaVectorStore

        # Initialize with test collection
        store = ChromaVectorStore(
            persist_directory="data/chroma_db_test",
            collection_name="test_collection",
            embedding_model="all-mpnet-base-v2",
            device="cpu",
        )

        console.print("[green]✓ ChromaDB initialized[/green]")

        # Add test documents
        test_docs = [
            "A contract is a legally binding agreement.",
            "Copyright protects original works.",
            "A trademark identifies products or services.",
        ]

        test_metadata = [
            {"source": "test1.txt", "type": "definition"},
            {"source": "test2.txt", "type": "definition"},
            {"source": "test3.txt", "type": "definition"},
        ]

        store.add_documents(documents=test_docs, metadatas=test_metadata)
        console.print(f"[green]✓ Added {len(test_docs)} test documents[/green]")

        # Test search
        results = store.search("contract definition", top_k=2)
        console.print(f"[green]✓ Search works: {len(results)} results[/green]")

        # Clean up test collection
        store.delete_collection()
        console.print("[green]✓ Cleanup successful[/green]")

        return True

    except Exception as e:
        console.print(f"[red]✗ ChromaDB test failed: {e}[/red]")
        return False


def test_query_router():
    """Test query router."""
    console.print("\n[bold cyan]Testing Query Router...[/bold cyan]")

    try:
        from src.agents.query_router import QueryRouter

        # Test rule-based routing
        router = QueryRouter(use_llm_classification=False)
        console.print("[green]✓ Query router initialized[/green]")

        test_queries = [
            ("What is a contract?", "simple"),
            ("Define non-disclosure agreement", "simple"),
            ("Analyze the implications of copyright", "complex"),
        ]

        for query, expected in test_queries:
            route = router.route(query)
            status = "✓" if route == expected else "⚠"
            console.print(f"[green]{status} '{query[:30]}...' → {route}[/green]")

        return True

    except Exception as e:
        console.print(f"[red]✗ Query router test failed: {e}[/red]")
        return False


def test_integration():
    """Test integrated system."""
    console.print("\n[bold cyan]Testing Integrated System...[/bold cyan]")

    try:
        from src.llm.ollama_client import OllamaClient
        from src.agents.unified_agent import UnifiedLegalAgent

        # Check if Ollama has the model
        client = OllamaClient(model="llama3.1:8b")
        if not client.test_connection():
            console.print(
                "[yellow]⚠ Ollama not available, skipping integration test[/yellow]"
            )
            return True

        models = client.list_models()
        if "llama3.1:8b" not in " ".join(models):
            console.print(
                "[yellow]⚠ llama3.1:8b not available, skipping integration test[/yellow]"
            )
            return True

        # Initialize agent (without RAG for quick test)
        agent = UnifiedLegalAgent(llm_client=client, enable_safety_filter=True)
        console.print("[green]✓ Unified agent initialized[/green]")

        # Test simple query
        console.print("Testing query processing...")
        response = agent.query("What is a contract in one sentence?")

        if response and len(response) > 20:
            console.print(
                f"[green]✓ Agent response received ({len(response)} chars)[/green]"
            )
            console.print(f"\n[dim]Response preview: {response[:100]}...[/dim]\n")
            return True
        else:
            console.print("[yellow]⚠ Agent response seems short[/yellow]")
            return False

    except Exception as e:
        console.print(f"[red]✗ Integration test failed: {e}[/red]")
        return False


def main():
    """Run all tests."""
    console.print(
        Panel.fit(
            "[bold cyan]Local LLM Setup Verification[/bold cyan]\n"
            "Testing all components...",
            border_style="cyan",
        )
    )

    results = {
        "Ollama": test_ollama(),
        "Embeddings": test_embeddings(),
        "ChromaDB": test_chroma(),
        "Query Router": test_query_router(),
        "Integration": test_integration(),
    }

    # Summary
    console.print("\n" + "=" * 60)
    console.print("[bold cyan]Test Summary[/bold cyan]")
    console.print("=" * 60)

    all_passed = True
    for component, passed in results.items():
        status = "[green]✓ PASS[/green]" if passed else "[red]✗ FAIL[/red]"
        console.print(f"{component:20} {status}")
        if not passed:
            all_passed = False

    console.print("=" * 60)

    if all_passed:
        console.print("\n[bold green]✓ All tests passed! Setup is ready.[/bold green]")
        console.print("\n[cyan]Next steps:[/cyan]")
        console.print("1. Add documents to data/documents/")
        console.print("2. Run: python scripts/ingest_documents.py")
        console.print("3. Start using the agent!")
        return 0
    else:
        console.print(
            "\n[bold red]✗ Some tests failed. Please check the errors above.[/bold red]"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
