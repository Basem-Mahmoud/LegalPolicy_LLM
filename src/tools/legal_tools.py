"""
Legal tools for function calling.
Provides utilities for document search, definitions, and legal references.
"""

import json
from typing import Dict, Any, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LegalTools:
    """
    Collection of tools that can be called by the LLM for enhanced functionality.
    """

    def __init__(self, vector_store=None):
        """
        Initialize legal tools.

        Args:
            vector_store: Optional VectorStore instance for document search
        """
        self.vector_store = vector_store
        self.legal_definitions = self._load_legal_definitions()

    def _load_legal_definitions(self) -> Dict[str, str]:
        """Load common legal term definitions."""
        return {
            "contract": "A legally binding agreement between two or more parties that creates mutual obligations enforceable by law.",
            "tort": "A civil wrong that causes harm or loss to another person, resulting in legal liability for the person who commits the tortious act.",
            "liability": "Legal responsibility for one's acts or omissions, especially for damages or injury caused to another party.",
            "jurisdiction": "The official power to make legal decisions and judgments, or the geographical area within which such power may be exercised.",
            "statute": "A written law passed by a legislative body, such as Congress or a state legislature.",
            "plaintiff": "A person who brings a case against another in a court of law.",
            "defendant": "A person or entity accused of a crime in criminal prosecution or against whom a civil claim is made.",
            "precedent": "A legal principle or rule established in a previous court decision that is binding or persuasive for courts when deciding subsequent cases with similar issues or facts.",
            "due process": "Fair treatment through the normal judicial system, especially as a citizen's entitlement under the Fifth and Fourteenth Amendments.",
            "discovery": "The pre-trial phase in a lawsuit in which each party investigates the facts of a case through depositions, interrogatories, and requests for documents.",
            "injunction": "A court order requiring a person or entity to do or refrain from doing specific acts.",
            "arbitration": "The use of an arbitrator to settle a dispute, as an alternative to litigation in court.",
            "indemnity": "A contractual obligation by one party to compensate the loss incurred by another party due to the acts of the indemnitor or another party.",
            "force majeure": "Unforeseeable circumstances that prevent someone from fulfilling a contract, such as natural disasters or war.",
            "nda": "Non-Disclosure Agreement - A legal contract that creates a confidential relationship between parties to protect sensitive information.",
        }

    def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """
        Get tool definitions for LLM function calling (OpenAI format).

        Returns:
            List of tool definition dictionaries
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": "search_legal_documents",
                    "description": "Search through the legal document database for relevant information. Use this when the user asks about specific policies, regulations, or legal documents.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant legal documents"
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of documents to retrieve (default: 3)",
                                "default": 3
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_legal_definition",
                    "description": "Get the definition of a legal term. Use this when the user asks about the meaning of a specific legal term or concept.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "term": {
                                "type": "string",
                                "description": "The legal term to define (e.g., 'contract', 'liability', 'tort')"
                            }
                        },
                        "required": ["term"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_statute_reference",
                    "description": "Look up information about a specific statute or regulation by its reference number. Use this when the user mentions a specific law by its code or number.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "statute_reference": {
                                "type": "string",
                                "description": "The statute reference (e.g., 'USC Title 17', 'EU GDPR Article 6')"
                            }
                        },
                        "required": ["statute_reference"]
                    }
                }
            }
        ]

    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> str:
        """
        Execute a tool by name with given arguments.

        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of tool arguments

        Returns:
            Tool execution result as string
        """
        tool_map = {
            "search_legal_documents": self.search_legal_documents,
            "get_legal_definition": self.get_legal_definition,
            "check_statute_reference": self.check_statute_reference,
        }

        if tool_name not in tool_map:
            return f"Error: Unknown tool '{tool_name}'"

        try:
            result = tool_map[tool_name](**arguments)
            return result
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {str(e)}")
            return f"Error executing tool: {str(e)}"

    def search_legal_documents(self, query: str, top_k: int = 3) -> str:
        """
        Search legal documents using vector store.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            Formatted search results
        """
        if not self.vector_store:
            return "Document search is not available. Vector store not initialized."

        # Convert top_k to int if it's a string (from LLM tool calls)
        if isinstance(top_k, str):
            try:
                top_k = int(top_k)
            except ValueError:
                top_k = 3  # Default fallback

        logger.info(f"Searching documents for: {query}")

        results = self.vector_store.search(query, top_k=top_k)

        if not results:
            return "No relevant documents found for your query."

        # Format results
        output_parts = [f"Found {len(results)} relevant documents:\n"]

        for i, result in enumerate(results, 1):
            source = result['metadata'].get('source', 'Unknown')
            content = result['content'][:300]  # First 300 chars

            output_parts.append(f"\n{i}. From '{source}':")
            output_parts.append(f"   {content}...")

        return "\n".join(output_parts)

    def get_legal_definition(self, term: str) -> str:
        """
        Get definition of a legal term.

        Args:
            term: Legal term to define

        Returns:
            Definition of the term
        """
        term_lower = term.lower().strip()

        logger.info(f"Looking up definition for: {term}")

        if term_lower in self.legal_definitions:
            definition = self.legal_definitions[term_lower]
            return f"**{term.title()}**: {definition}"
        else:
            return f"I don't have a specific definition for '{term}' in my database. I can provide a general explanation based on my knowledge, or you can search the legal documents for more information."

    def check_statute_reference(self, statute_reference: str) -> str:
        """
        Look up a statute reference (mock implementation).

        Args:
            statute_reference: Statute reference code

        Returns:
            Information about the statute
        """
        logger.info(f"Looking up statute: {statute_reference}")

        # This is a mock implementation
        # In a real system, this would query a legal database or API

        mock_statutes = {
            "usc title 17": "United States Code Title 17 - Copyright Law. Governs copyright protection in the United States.",
            "gdpr article 6": "EU General Data Protection Regulation Article 6 - Lawfulness of processing. Defines the legal bases for processing personal data.",
            "cfr title 21": "Code of Federal Regulations Title 21 - Food and Drugs. Regulations for food, drugs, and medical devices.",
        }

        ref_lower = statute_reference.lower().strip()

        # Try to find a match
        for key, description in mock_statutes.items():
            if key in ref_lower or ref_lower in key:
                return f"**{statute_reference}**: {description}\n\nNote: This is general information. Consult the actual statute text and legal counsel for specific matters."

        return f"I don't have specific information about '{statute_reference}' in my database. For detailed statute information, please consult official legal databases or a qualified attorney."


def format_tool_response_for_llm(tool_name: str, tool_result: str) -> str:
    """
    Format tool response for inclusion in LLM context.

    Args:
        tool_name: Name of the tool that was called
        tool_result: Result from the tool

    Returns:
        Formatted response
    """
    return f"""Tool Used: {tool_name}

Result:
{tool_result}

Use this information to provide a comprehensive answer to the user's question."""


if __name__ == "__main__":
    # Example usage
    tools = LegalTools()

    # Test definition lookup
    print("Testing get_legal_definition:")
    print(tools.get_legal_definition("contract"))
    print("\n")

    # Test statute lookup
    print("Testing check_statute_reference:")
    print(tools.check_statute_reference("GDPR Article 6"))
    print("\n")

    # Get tool definitions for LLM
    print("Tool definitions for LLM:")
    print(json.dumps(tools.get_tool_definitions(), indent=2))
