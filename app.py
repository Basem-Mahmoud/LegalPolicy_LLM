"""
Main application file for the Legal Policy Explainer.
Provides CLI interface for interacting with the assistant.
"""

import os
import sys
from pathlib import Path
from typing import Optional
import logging
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.llm_client import LLMClient
from src.rag import DocumentProcessor, VectorStore, RAGRetriever
from src.agents import MultiAgentOrchestrator, SafetyFilter
from src.tools import LegalTools
from src.prompts import get_system_prompt, add_disclaimer

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class LegalPolicyExplainer:
    """
    Main Legal Policy Explainer application.
    """

    def __init__(
        self,
        use_rag: bool = True,
        use_multi_agent: bool = True,
        use_tools: bool = True
    ):
        """
        Initialize the Legal Policy Explainer.

        Args:
            use_rag: Whether to use RAG for document retrieval
            use_multi_agent: Whether to use multi-agent system
            use_tools: Whether to enable tool calling
        """
        logger.info("Initializing Legal Policy Explainer...")

        self.use_rag = use_rag
        self.use_multi_agent = use_multi_agent
        self.use_tools = use_tools

        # Initialize LLM client
        logger.info("Loading LLM client...")
        self.llm_client = LLMClient()

        # Initialize RAG system
        self.rag_retriever = None
        if use_rag:
            logger.info("Initializing RAG system...")
            self.rag_retriever = self._initialize_rag()

        # Initialize tools
        self.tools = None
        if use_tools:
            logger.info("Initializing tools...")
            vector_store = self.rag_retriever.vector_store if self.rag_retriever else None
            self.tools = LegalTools(vector_store=vector_store)

        # Initialize multi-agent system or simple assistant
        if use_multi_agent:
            logger.info("Initializing multi-agent system...")
            self.orchestrator = MultiAgentOrchestrator(
                llm_client=self.llm_client,
                rag_retriever=self.rag_retriever,
                tools=self.tools
            )
        else:
            self.orchestrator = None

        # Initialize safety filter
        self.safety_filter = SafetyFilter()

        logger.info("Legal Policy Explainer initialized successfully!")

    def _initialize_rag(self) -> Optional[RAGRetriever]:
        """
        Initialize RAG system with document processing and vector store.

        Returns:
            RAGRetriever instance or None
        """
        try:
            # Check if vector database exists
            vector_store = VectorStore()

            # If empty, process and add documents
            if vector_store.get_stats()["document_count"] == 0:
                logger.info("Vector store is empty. Processing documents...")
                doc_processor = DocumentProcessor()
                documents = doc_processor.load_documents()

                if documents:
                    vector_store.add_documents(documents)
                    logger.info(f"Added {len(documents)} documents to vector store")
                else:
                    logger.warning("No documents found to process")

            # Create retriever
            retriever = RAGRetriever(vector_store=vector_store)
            return retriever

        except Exception as e:
            logger.error(f"Error initializing RAG: {str(e)}")
            return None

    def query(self, user_query: str) -> str:
        """
        Process a user query and return response.

        Args:
            user_query: User's question

        Returns:
            Assistant's response
        """
        # Safety check
        safety_check = self.safety_filter.check_query(user_query)

        if not safety_check["allowed"]:
            return safety_check["message"]

        # Add caution message if needed
        caution_message = ""
        if safety_check.get("caution"):
            caution_message = f"\n\n⚠️ {safety_check['message']}\n"

        # Generate response
        try:
            if self.use_multi_agent and self.orchestrator:
                # Use multi-agent system
                response = self.orchestrator.get_explanation(user_query)
            else:
                # Use simple single-agent approach
                response = self._simple_query(user_query)

            return caution_message + response

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return f"I apologize, but I encountered an error processing your question. Please try again or rephrase your query.\n\nError: {str(e)}"

    def _simple_query(self, user_query: str) -> str:
        """
        Simple query without multi-agent system.

        Args:
            user_query: User's question

        Returns:
            Response
        """
        # Get system prompt
        system_prompt = get_system_prompt(role="explainer", include_examples=True)

        # Retrieve context if RAG is enabled
        context = ""
        if self.rag_retriever:
            docs, context = self.rag_retriever.retrieve_and_format(user_query)

        # Build prompt
        if context:
            full_prompt = f"{context}\n\nUser question: {user_query}"
        else:
            full_prompt = user_query

        # Generate response
        response = self.llm_client.generate_with_system_prompt(
            system_prompt=system_prompt,
            user_message=full_prompt
        )

        # Add disclaimer
        response = add_disclaimer(response)

        return response

    def interactive_mode(self):
        """Run the assistant in interactive CLI mode."""
        print("\n" + "="*80)
        print("Legal Policy Explainer - Interactive Mode")
        print("="*80)
        print("\n⚠️ DISCLAIMER: This assistant provides general information only,")
        print("not legal advice. Always consult a qualified attorney for specific legal matters.\n")
        print("Type 'exit' or 'quit' to end the session.\n")
        print("="*80 + "\n")

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\nThank you for using Legal Policy Explainer. Goodbye!")
                    break

                # Process query
                print("\nAssistant: ", end="", flush=True)
                response = self.query(user_input)
                print(response)

            except KeyboardInterrupt:
                print("\n\nSession interrupted. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Error in interactive mode: {str(e)}")
                print(f"\nError: {str(e)}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Legal Policy Explainer Assistant")
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG system"
    )
    parser.add_argument(
        "--no-multi-agent",
        action="store_true",
        help="Disable multi-agent system"
    )
    parser.add_argument(
        "--no-tools",
        action="store_true",
        help="Disable tool calling"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single query mode (non-interactive)"
    )

    args = parser.parse_args()

    # Initialize assistant
    assistant = LegalPolicyExplainer(
        use_rag=not args.no_rag,
        use_multi_agent=not args.no_multi_agent,
        use_tools=not args.no_tools
    )

    # Single query mode or interactive mode
    if args.query:
        response = assistant.query(args.query)
        print("\n" + response)
    else:
        assistant.interactive_mode()


if __name__ == "__main__":
    main()
