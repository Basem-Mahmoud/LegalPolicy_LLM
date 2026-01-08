"""
Multi-agent system for legal policy explanation.
Implements Researcher and Explainer agents that collaborate to provide comprehensive answers.
"""

from typing import Dict, Any, List, Optional
import logging

from ..llm_client import LLMClient
from ..prompts import (
    RESEARCHER_AGENT_PROMPT,
    EXPLAINER_AGENT_PROMPT,
    add_disclaimer
)
from ..rag import RAGRetriever
from ..tools import LegalTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Agent:
    """Base agent class."""

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        llm_client: LLMClient,
        temperature: float = 0.3
    ):
        """
        Initialize agent.

        Args:
            name: Agent name
            role: Agent role description
            system_prompt: System prompt for the agent
            llm_client: LLM client instance
            temperature: Generation temperature
        """
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.temperature = temperature
        self.conversation_history = []

    def generate(self, user_message: str, context: Optional[str] = None) -> str:
        """
        Generate response from agent.

        Args:
            user_message: User query
            context: Optional context to include

        Returns:
            Agent response
        """
        # Prepare messages
        messages = [{"role": "system", "content": self.system_prompt}]

        # Add context if provided
        if context:
            messages.append({
                "role": "system",
                "content": f"Context:\n{context}"
            })

        # Add conversation history
        messages.extend(self.conversation_history)

        # Add current message
        messages.append({"role": "user", "content": user_message})

        # Generate response
        response = self.llm_client.generate(
            messages=messages,
            temperature=self.temperature
        )

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []


class ResearcherAgent(Agent):
    """
    Researcher agent that retrieves and analyzes legal documents.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        rag_retriever: Optional[RAGRetriever] = None,
        tools: Optional[LegalTools] = None
    ):
        """
        Initialize researcher agent.

        Args:
            llm_client: LLM client instance
            rag_retriever: RAG retriever for document search
            tools: Legal tools instance
        """
        super().__init__(
            name="Researcher",
            role="Legal Document Researcher",
            system_prompt=RESEARCHER_AGENT_PROMPT,
            llm_client=llm_client,
            temperature=0.2  # Lower temperature for factual research
        )
        self.rag_retriever = rag_retriever
        self.tools = tools

    def research(self, query: str) -> Dict[str, Any]:
        """
        Research a query by retrieving relevant documents and extracting key information.

        Args:
            query: User query to research

        Returns:
            Dictionary with research findings
        """
        logger.info(f"Researcher agent researching: {query}")

        findings = {
            "query": query,
            "retrieved_documents": [],
            "key_findings": "",
            "context": ""
        }

        # Retrieve relevant documents
        if self.rag_retriever:
            docs, context = self.rag_retriever.retrieve_and_format(query)
            findings["retrieved_documents"] = docs
            findings["context"] = context

            # Ask researcher to analyze the documents
            analysis_prompt = f"""Analyze the following documents to answer the query: "{query}"

{context}

Provide:
1. Key findings relevant to the query
2. Important legal terms that need explanation
3. Any caveats or limitations"""

            analysis = self.generate(analysis_prompt)
            findings["key_findings"] = analysis

        else:
            # No RAG available, just use general knowledge
            research_prompt = f"""Research and provide key information about: {query}

Focus on:
1. Main concepts and definitions
2. Important legal principles
3. Common applications or examples
4. Terms that need explanation"""

            research = self.generate(research_prompt)
            findings["key_findings"] = research

        return findings


class ExplainerAgent(Agent):
    """
    Explainer agent that translates legal information into clear explanations.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Initialize explainer agent.

        Args:
            llm_client: LLM client instance
        """
        super().__init__(
            name="Explainer",
            role="Legal Policy Explainer",
            system_prompt=EXPLAINER_AGENT_PROMPT,
            llm_client=llm_client,
            temperature=0.4  # Slightly higher for more natural explanations
        )

    def explain(self, query: str, research_findings: Dict[str, Any]) -> str:
        """
        Generate clear explanation based on research findings.

        Args:
            query: Original user query
            research_findings: Research findings from ResearcherAgent

        Returns:
            Clear, accessible explanation
        """
        logger.info(f"Explainer agent explaining: {query}")

        # Prepare explanation prompt
        explanation_prompt = f"""The user asked: "{query}"

Research findings:
{research_findings.get('key_findings', 'No specific findings available.')}

{research_findings.get('context', '')}

Please provide a clear, accessible explanation that:
1. Answers the user's question directly
2. Explains any legal jargon in simple terms
3. Uses examples or analogies where helpful
4. Is structured and easy to follow

Remember to be educational and informative, not to provide legal advice."""

        explanation = self.generate(explanation_prompt)

        # Add disclaimer
        explanation_with_disclaimer = add_disclaimer(explanation)

        return explanation_with_disclaimer


class MultiAgentOrchestrator:
    """
    Orchestrator that coordinates Researcher and Explainer agents.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        rag_retriever: Optional[RAGRetriever] = None,
        tools: Optional[LegalTools] = None,
        max_iterations: int = 1
    ):
        """
        Initialize multi-agent orchestrator.

        Args:
            llm_client: LLM client instance
            rag_retriever: RAG retriever for document search
            tools: Legal tools instance
            max_iterations: Maximum refinement iterations
        """
        self.llm_client = llm_client
        self.max_iterations = max_iterations

        # Initialize agents
        self.researcher = ResearcherAgent(
            llm_client=llm_client,
            rag_retriever=rag_retriever,
            tools=tools
        )
        self.explainer = ExplainerAgent(llm_client=llm_client)

        logger.info("Multi-agent system initialized")

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a query using multi-agent collaboration.

        Args:
            query: User query

        Returns:
            Dictionary with complete response and metadata
        """
        logger.info(f"Processing query with multi-agent system: {query}")

        # Step 1: Research phase
        research_findings = self.researcher.research(query)

        # Step 2: Explanation phase
        explanation = self.explainer.explain(query, research_findings)

        # Compile response
        response = {
            "query": query,
            "explanation": explanation,
            "research_findings": research_findings,
            "retrieved_documents": research_findings.get("retrieved_documents", []),
            "agents_used": ["Researcher", "Explainer"]
        }

        return response

    def get_explanation(self, query: str) -> str:
        """
        Get explanation for a query (simplified interface).

        Args:
            query: User query

        Returns:
            Explanation string
        """
        response = self.process_query(query)
        return response["explanation"]

    def reset_agents(self):
        """Reset all agents' conversation history."""
        self.researcher.reset_conversation()
        self.explainer.reset_conversation()


class SafetyFilter:
    """
    Safety filter to check for inappropriate queries.
    """

    def __init__(self):
        """Initialize safety filter."""
        self.blocked_patterns = [
            "how to break the law",
            "how to avoid",
            "loophole",
            "get away with",
            "evade",
            "circumvent the law"
        ]

        self.caution_patterns = [
            "my case",
            "my lawsuit",
            "should i sue",
            "what will happen in my",
            "will i win",
            "predict the outcome"
        ]

    def check_query(self, query: str) -> Dict[str, Any]:
        """
        Check if query should be blocked or flagged.

        Args:
            query: User query

        Returns:
            Dictionary with safety check results
        """
        query_lower = query.lower()

        # Check for blocked patterns
        for pattern in self.blocked_patterns:
            if pattern in query_lower:
                return {
                    "allowed": False,
                    "reason": "inappropriate_request",
                    "message": "I cannot provide guidance on circumventing or violating laws. If you have legal concerns, please consult a qualified attorney."
                }

        # Check for caution patterns
        for pattern in self.caution_patterns:
            if pattern in query_lower:
                return {
                    "allowed": True,
                    "caution": True,
                    "reason": "specific_legal_advice",
                    "message": "I can provide general information, but I cannot give advice on your specific legal situation. Please consult a qualified attorney for personalized legal advice."
                }

        return {
            "allowed": True,
            "caution": False
        }


if __name__ == "__main__":
    # Example usage
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize system
    llm_client = LLMClient()
    orchestrator = MultiAgentOrchestrator(llm_client=llm_client)

    # Test query
    query = "What is a non-disclosure agreement?"
    response = orchestrator.process_query(query)

    print("="*80)
    print("QUERY:", query)
    print("="*80)
    print("\nEXPLANATION:")
    print(response["explanation"])
    print("="*80)
