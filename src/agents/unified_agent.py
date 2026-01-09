"""
Unified Legal Agent using LangGraph.
Single agent that handles all query types with intelligent routing.
Replaces the old multi-agent system (Researcher + Explainer) for better performance.
"""

import logging
from typing import TypedDict, Annotated, Literal, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
import operator

from .query_router import QueryRouter, QueryComplexity
from ..llm.ollama_client import OllamaClient
from ..rag.rag_retriever import RAGRetriever
from ..tools.legal_tools import LegalTools
from ..prompts.system_prompts import UNIFIED_AGENT_PROMPT

logger = logging.getLogger(__name__)


# Define the agent state
class AgentState(TypedDict):
    """State that flows through the agent graph."""

    query: str  # Original user query
    complexity: Optional[str]  # Query complexity: simple, medium, complex
    requires_rag: bool  # Whether RAG is needed
    requires_tools: bool  # Whether tools are needed
    retrieved_context: Optional[str]  # Retrieved documents context
    tool_results: Annotated[List[Dict[str, Any]], operator.add]  # Tool call results
    response: Optional[str]  # Final response
    is_safe: bool  # Whether query passed safety filter
    disclaimer_added: bool  # Whether disclaimer was added


class UnifiedLegalAgent:
    """
    Unified agent that handles all legal queries using LangGraph.
    Routes queries intelligently and processes them efficiently.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        rag_retriever: Optional[RAGRetriever] = None,
        query_router: Optional[QueryRouter] = None,
        enable_safety_filter: bool = True,
        enable_tools: bool = True,
    ):
        """
        Initialize unified legal agent.

        Args:
            llm_client: OllamaClient instance
            rag_retriever: Optional RAGRetriever for document retrieval
            query_router: Optional QueryRouter for query classification
            enable_safety_filter: Whether to enable safety filtering
            enable_tools: Whether to enable tool calling
        """
        self.llm_client = llm_client
        self.rag_retriever = rag_retriever
        self.enable_safety_filter = enable_safety_filter
        self.enable_tools = enable_tools

        # Initialize query router
        self.query_router = query_router or QueryRouter(
            llm_client=llm_client, use_llm_classification=True
        )

        # Initialize legal tools
        vector_store = rag_retriever.vector_store if rag_retriever else None
        self.legal_tools = LegalTools(vector_store=vector_store)

        # Build the agent graph
        self.graph = self._build_graph()

        logger.info("UnifiedLegalAgent initialized with LangGraph")

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("safety_filter", self._safety_filter)
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("simple_path", self._simple_path)
        workflow.add_node("medium_path", self._medium_path)
        workflow.add_node("complex_path", self._complex_path)
        workflow.add_node("add_disclaimer", self._add_disclaimer)

        # Define edges
        workflow.set_entry_point("safety_filter")

        # Safety filter conditional edge
        workflow.add_conditional_edges(
            "safety_filter",
            self._check_safety,
            {
                "safe": "route_query",
                "unsafe": "add_disclaimer",  # Go straight to disclaimer with rejection
            },
        )

        # Route query based on complexity
        workflow.add_conditional_edges(
            "route_query",
            self._get_route,
            {
                "simple": "simple_path",
                "medium": "medium_path",
                "complex": "complex_path",
            },
        )

        # All paths lead to disclaimer
        workflow.add_edge("simple_path", "add_disclaimer")
        workflow.add_edge("medium_path", "add_disclaimer")
        workflow.add_edge("complex_path", "add_disclaimer")

        # Disclaimer is the final node
        workflow.add_edge("add_disclaimer", END)

        return workflow.compile()

    def _safety_filter(self, state: AgentState) -> AgentState:
        """
        Filter out inappropriate queries.

        Args:
            state: Current agent state

        Returns:
            Updated state with safety check
        """
        query = state["query"]
        logger.info("Running safety filter")

        if not self.enable_safety_filter:
            state["is_safe"] = True
            return state

        # Define blocked topics/patterns
        blocked_patterns = [
            "personal legal advice",
            "specific case advice",
            "should i sue",
            "what should i do legally",
            "medical advice",
            "financial advice",
        ]

        query_lower = query.lower()
        is_blocked = any(pattern in query_lower for pattern in blocked_patterns)

        if is_blocked:
            state["is_safe"] = False
            state["response"] = (
                "I cannot provide personal legal advice or recommendations for specific legal actions. "
                "Please consult with a licensed attorney for guidance on your specific situation."
            )
            logger.warning(f"Query blocked by safety filter: {query[:50]}...")
        else:
            state["is_safe"] = True

        return state

    def _check_safety(self, state: AgentState) -> Literal["safe", "unsafe"]:
        """Check if query is safe."""
        return "safe" if state["is_safe"] else "unsafe"

    def _route_query(self, state: AgentState) -> AgentState:
        """
        Route query based on complexity.

        Args:
            state: Current agent state

        Returns:
            Updated state with routing information
        """
        query = state["query"]
        classification = self.query_router.classify_query(query)

        state["complexity"] = classification.complexity.value
        state["requires_rag"] = classification.requires_rag
        state["requires_tools"] = classification.requires_tools

        logger.info(
            f"Query routed to {classification.complexity} path (RAG: {classification.requires_rag}, Tools: {classification.requires_tools})"
        )

        return state

    def _get_route(self, state: AgentState) -> Literal["simple", "medium", "complex"]:
        """Get the route destination based on complexity."""
        return state["complexity"]

    def _simple_path(self, state: AgentState) -> AgentState:
        """
        Handle simple queries (definitions, basic explanations).
        No RAG or tools needed.

        Args:
            state: Current agent state

        Returns:
            Updated state with response
        """
        query = state["query"]
        logger.info("Processing via simple path (no RAG)")

        messages = [
            {"role": "system", "content": UNIFIED_AGENT_PROMPT},
            {
                "role": "user",
                "content": f"Provide a clear, concise explanation for this query:\n\n{query}",
            },
        ]

        response = self.llm_client.generate(messages, temperature=0.3)
        state["response"] = response

        return state

    def _medium_path(self, state: AgentState) -> AgentState:
        """
        Handle medium complexity queries with RAG retrieval.

        Args:
            state: Current agent state

        Returns:
            Updated state with response
        """
        query = state["query"]
        logger.info("Processing via medium path (with RAG)")

        # Retrieve relevant context
        if self.rag_retriever:
            docs, context = self.rag_retriever.retrieve_and_format(query)
            state["retrieved_context"] = context
            logger.info(f"Retrieved {len(docs)} documents for context")
        else:
            context = ""
            state["retrieved_context"] = ""
            logger.warning("RAG retriever not available, proceeding without context")

        # Generate response with context
        if context:
            user_message = f"""Based on the following legal documents, answer the user's question.

{context}

User Question: {query}

Provide a clear, accurate explanation based on the documents above."""
        else:
            user_message = f"Answer this legal question clearly and accurately:\n\n{query}"

        messages = [
            {"role": "system", "content": UNIFIED_AGENT_PROMPT},
            {"role": "user", "content": user_message},
        ]

        response = self.llm_client.generate(messages, temperature=0.3)
        state["response"] = response

        return state

    def _complex_path(self, state: AgentState) -> AgentState:
        """
        Handle complex queries with RAG + tool calling.

        Args:
            state: Current agent state

        Returns:
            Updated state with response
        """
        query = state["query"]
        logger.info("Processing via complex path (RAG + tools)")

        # Retrieve context (same as medium path)
        if self.rag_retriever:
            docs, context = self.rag_retriever.retrieve_and_format(query)
            state["retrieved_context"] = context
            logger.info(f"Retrieved {len(docs)} documents for context")
        else:
            context = ""
            state["retrieved_context"] = ""

        # Tool calling logic
        tool_context = ""
        if self.enable_tools and state["requires_tools"]:
            tool_context = self._execute_tools(query, state)

        # Build comprehensive prompt with all context
        context_parts = []

        if context:
            context_parts.append(f"=== Legal Documents Context ===\n{context}")

        if tool_context:
            context_parts.append(f"\n=== Tool Results ===\n{tool_context}")

        if context_parts:
            user_message = f"""{chr(10).join(context_parts)}

User Question: {query}

Provide a detailed, well-structured analysis that:
1. Addresses all aspects of the question
2. Cites relevant information from the documents and tool results
3. Explains implications and connections
4. Uses clear legal reasoning"""
        else:
            user_message = f"Provide a comprehensive legal analysis for:\n\n{query}"

        messages = [
            {"role": "system", "content": UNIFIED_AGENT_PROMPT},
            {"role": "user", "content": user_message},
        ]

        response = self.llm_client.generate(messages, temperature=0.4, max_tokens=3000)
        state["response"] = response

        return state

    def _execute_tools(self, query: str, state: AgentState) -> str:
        """
        Execute tools based on LLM's tool calling decisions.

        Args:
            query: User's query
            state: Current agent state

        Returns:
            Formatted tool results
        """
        logger.info("Initiating tool calling")

        try:
            # Get tool definitions
            tools = self.legal_tools.get_tool_definitions()

            # Prepare messages for tool calling
            messages = [
                {"role": "system", "content": UNIFIED_AGENT_PROMPT},
                {
                    "role": "user",
                    "content": f"""Analyze this legal query and determine which tools (if any) should be called to answer it comprehensively.

Query: {query}

You have access to these tools:
1. search_legal_documents - Search the legal document database
2. get_legal_definition - Get definitions of legal terms
3. check_statute_reference - Look up specific statutes

Decide which tools to call based on the query.""",
                },
            ]

            # Call LLM with tools
            response = self.llm_client.generate_with_tools(
                messages=messages, tools=tools, temperature=0.3
            )

            # Check if LLM requested tool calls
            if "message" in response and "tool_calls" in response["message"]:
                tool_calls = response["message"]["tool_calls"]

                if not tool_calls:
                    logger.info("No tools called by LLM")
                    return ""

                logger.info(f"LLM requested {len(tool_calls)} tool call(s)")

                # Execute each tool call
                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call["function"]["name"]
                    tool_args = tool_call["function"]["arguments"]

                    logger.info(f"Executing tool: {tool_name} with args: {tool_args}")

                    # Execute the tool
                    result = self.legal_tools.execute_tool(tool_name, tool_args)

                    # Store result
                    tool_result = {
                        "tool": tool_name,
                        "arguments": tool_args,
                        "result": result,
                    }
                    tool_results.append(tool_result)
                    state["tool_results"].append(tool_result)

                # Format tool results for LLM context
                formatted_results = []
                for tr in tool_results:
                    formatted_results.append(
                        f"Tool: {tr['tool']}\nArguments: {tr['arguments']}\nResult:\n{tr['result']}\n"
                    )

                return "\n---\n".join(formatted_results)

            else:
                logger.info("LLM response did not include tool calls")
                return ""

        except Exception as e:
            logger.error(f"Error in tool execution: {e}")
            # Don't fail the entire query if tools fail
            return f"Note: Tool execution encountered an error: {str(e)}"

    def _add_disclaimer(self, state: AgentState) -> AgentState:
        """
        Add legal disclaimer to response.

        Args:
            state: Current agent state

        Returns:
            Updated state with disclaimer
        """
        if state.get("response"):
            disclaimer = (
                "\n\n⚖️ Disclaimer: This is an educational tool and not a substitute for professional legal advice. "
                "Consult a licensed attorney for legal matters specific to your situation."
            )
            state["response"] = state["response"] + disclaimer
            state["disclaimer_added"] = True
            logger.info("Disclaimer added to response")

        return state

    def query(self, user_query: str) -> str:
        """
        Process a user query through the unified agent.

        Args:
            user_query: User's question

        Returns:
            Agent's response with disclaimer
        """
        logger.info(f"Processing query: {user_query[:50]}...")

        # Initialize state
        initial_state: AgentState = {
            "query": user_query,
            "complexity": None,
            "requires_rag": False,
            "requires_tools": False,
            "retrieved_context": None,
            "tool_results": [],
            "response": None,
            "is_safe": True,
            "disclaimer_added": False,
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        response = final_state.get("response", "I apologize, but I couldn't generate a response.")
        logger.info("Query processing complete")

        return response

    def __repr__(self) -> str:
        return f"UnifiedLegalAgent(model={self.llm_client.model}, rag_enabled={self.rag_retriever is not None})"


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    print("=== Initializing Unified Legal Agent ===\n")

    # Initialize LLM client
    from src.llm.ollama_client import OllamaClient

    llm_client = OllamaClient(model="llama3.1:8b")

    if not llm_client.test_connection():
        print("❌ Ollama not running. Please start it with: ollama serve")
        exit(1)

    # Initialize agent (without RAG for basic testing)
    agent = UnifiedLegalAgent(llm_client=llm_client, enable_safety_filter=True)

    print("✓ Agent initialized\n")

    # Test queries
    test_queries = [
        "What is a contract?",  # Simple
        "What are the key elements of a valid contract?",  # Medium
        "Should I sue my neighbor?",  # Unsafe (should be blocked)
    ]

    for query in test_queries:
        print(f"\n{'=' * 80}")
        print(f"Query: {query}")
        print(f"{'=' * 80}\n")

        response = agent.query(query)
        print(response)
        print()
