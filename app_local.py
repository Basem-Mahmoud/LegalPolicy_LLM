"""
Main application for Local LLM Legal Policy Explainer.
100% local implementation with Ollama + ChromaDB + LangGraph.
"""

import argparse
import logging
import yaml
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.markdown import Markdown

from src.llm.ollama_client import OllamaClient
from src.rag.chroma_store import ChromaVectorStore
from src.rag.rag_retriever import RAGRetriever
from src.agents.unified_agent import UnifiedLegalAgent

console = Console()


def load_config(config_path: str = "config/config_local.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict) -> None:
    """Setup logging based on configuration."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))

    logging.basicConfig(
        level=level,
        format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        handlers=[
            logging.FileHandler(log_config.get("file", "logs/app.log")),
            logging.StreamHandler() if log_config.get("console_output", True) else logging.NullHandler()
        ]
    )


def initialize_system(config: dict, enable_rag: bool = True) -> UnifiedLegalAgent:
    """
    Initialize the legal assistant system.

    Args:
        config: Configuration dictionary
        enable_rag: Whether to enable RAG retrieval

    Returns:
        Initialized UnifiedLegalAgent
    """
    console.print("\n[bold cyan]Initializing Legal Policy Explainer (Local LLM)...[/bold cyan]\n")

    # Initialize LLM client
    console.print("1. Initializing Ollama client...")
    llm_config = config["llm"]
    llm_client = OllamaClient(
        model=llm_config["model"],
        base_url=llm_config["base_url"],
        temperature=llm_config["temperature"],
        max_tokens=llm_config["max_tokens"],
        timeout=llm_config["timeout"]
    )

    if not llm_client.test_connection():
        console.print("[red]✗ Cannot connect to Ollama[/red]")
        console.print("[yellow]Please start Ollama: ollama serve[/yellow]")
        raise RuntimeError("Ollama not available")

    console.print(f"   [green]✓ Connected to Ollama ({llm_config['model']})[/green]")

    # Initialize RAG components if enabled
    rag_retriever = None
    if enable_rag:
        console.print("2. Initializing RAG system...")

        # Initialize vector store
        vs_config = config["vector_store"]
        emb_config = config["embeddings"]

        vector_store = ChromaVectorStore(
            persist_directory=vs_config["persist_directory"],
            collection_name=vs_config["collection_name"],
            embedding_model=emb_config["model"],
            device=emb_config["device"],
            distance_metric=vs_config["distance_metric"]
        )

        doc_count = vector_store.collection.count()
        console.print(f"   [green]✓ ChromaDB loaded ({doc_count} documents)[/green]")

        if doc_count == 0:
            console.print("   [yellow]⚠ No documents in vector store[/yellow]")
            console.print("   [yellow]  Run: python scripts/ingest_documents.py[/yellow]")

        # Initialize RAG retriever
        rag_config = config["rag"]
        rag_retriever = RAGRetriever(
            vector_store=vector_store,
            top_k=rag_config["top_k"],
            similarity_threshold=rag_config["similarity_threshold"],
            max_context_length=rag_config["max_context_length"]
        )
        console.print("   [green]✓ RAG retriever initialized[/green]")
    else:
        console.print("2. RAG disabled (simple mode)")

    # Initialize unified agent
    console.print("3. Initializing unified agent...")
    agent_config = config["agent"]
    safety_config = config["safety"]

    agent = UnifiedLegalAgent(
        llm_client=llm_client,
        rag_retriever=rag_retriever,
        enable_safety_filter=safety_config["enable_filter"],
        enable_tools=agent_config["enable_tools"]
    )

    console.print("   [green]✓ Unified agent ready[/green]")
    console.print("\n[bold green]✓ System initialized successfully![/bold green]\n")

    return agent


def interactive_mode(agent: UnifiedLegalAgent) -> None:
    """
    Run interactive chat mode.

    Args:
        agent: Initialized UnifiedLegalAgent
    """
    console.print(Panel.fit(
        "[bold cyan]Legal Policy Explainer - Interactive Mode[/bold cyan]\n"
        "Ask questions about legal policies and regulations.\n"
        "Type 'quit' or 'exit' to stop.",
        border_style="cyan"
    ))

    console.print("\n[dim]Examples:[/dim]")
    console.print("[dim]  • What is a non-disclosure agreement?[/dim]")
    console.print("[dim]  • Explain the key elements of a valid contract.[/dim]")
    console.print("[dim]  • What are the implications of copyright infringement?[/dim]\n")

    while True:
        try:
            # Get user input
            query = Prompt.ask("\n[bold cyan]Your question[/bold cyan]")

            # Check for exit
            if query.lower() in ["quit", "exit", "q"]:
                console.print("\n[cyan]Thank you for using Legal Policy Explainer![/cyan]")
                break

            if not query.strip():
                continue

            # Process query
            console.print("\n[dim]Processing...[/dim]\n")
            response = agent.query(query)

            # Display response
            console.print(Panel(
                Markdown(response),
                title="[bold green]Response[/bold green]",
                border_style="green"
            ))

        except KeyboardInterrupt:
            console.print("\n\n[cyan]Interrupted. Goodbye![/cyan]")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]")
            logging.exception("Error processing query")


def single_query_mode(agent: UnifiedLegalAgent, query: str) -> None:
    """
    Process a single query and exit.

    Args:
        agent: Initialized UnifiedLegalAgent
        query: User query to process
    """
    console.print(f"\n[bold cyan]Query:[/bold cyan] {query}\n")
    console.print("[dim]Processing...[/dim]\n")

    response = agent.query(query)

    console.print(Panel(
        Markdown(response),
        title="[bold green]Response[/bold green]",
        border_style="green"
    ))


def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="Legal Policy Explainer with Local LLMs"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config_local.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query to process (skips interactive mode)"
    )
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG retrieval (use LLM knowledge only)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        console.print(f"[red]Error loading config: {e}[/red]")
        return 1

    # Setup logging
    if args.verbose:
        config["logging"]["level"] = "DEBUG"
    setup_logging(config)

    # Initialize system
    try:
        agent = initialize_system(config, enable_rag=not args.no_rag)
    except Exception as e:
        console.print(f"\n[red]Failed to initialize system: {e}[/red]")
        logging.exception("Initialization error")
        return 1

    # Run appropriate mode
    try:
        if args.query:
            # Single query mode
            single_query_mode(agent, args.query)
        else:
            # Interactive mode
            interactive_mode(agent)
    except Exception as e:
        console.print(f"\n[red]Runtime error: {e}[/red]")
        logging.exception("Runtime error")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
