"""
Query Router for intelligent routing based on query complexity.
Routes queries to appropriate processing paths:
- Simple: Direct definition lookup
- Medium: RAG-based retrieval
- Complex: Full pipeline with tool calling
"""

import logging
from typing import Literal, Optional, List
from pydantic import BaseModel, Field
from enum import Enum

logger = logging.getLogger(__name__)


class QueryComplexity(str, Enum):
    """Query complexity levels."""

    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class QueryClassification(BaseModel):
    """Structured output for query classification."""

    complexity: QueryComplexity = Field(
        description="Complexity level of the query: simple, medium, or complex"
    )
    reasoning: str = Field(
        description="Brief explanation of why this complexity was assigned"
    )
    requires_rag: bool = Field(
        description="Whether RAG retrieval is needed for this query"
    )
    requires_tools: bool = Field(
        description="Whether tool calling is needed for this query"
    )


class QueryRouter:
    """
    Intelligent query router that classifies queries by complexity
    and routes them to appropriate processing paths.
    """

    def __init__(
        self,
        llm_client=None,
        use_llm_classification: bool = True,
        simple_keywords: Optional[List[str]] = None,
        complex_keywords: Optional[List[str]] = None,
    ):
        """
        Initialize query router.

        Args:
            llm_client: Optional LLM client for intelligent classification
            use_llm_classification: Whether to use LLM for classification
            simple_keywords: Keywords indicating simple queries
            complex_keywords: Keywords indicating complex queries
        """
        self.llm_client = llm_client
        self.use_llm_classification = use_llm_classification and llm_client is not None

        # Default keyword patterns
        self.simple_keywords = simple_keywords or [
            "define",
            "what is",
            "meaning",
            "definition",
            "explain briefly",
            "in simple terms",
            "quick explanation",
        ]

        self.complex_keywords = complex_keywords or [
            "analyze",
            "compare",
            "evaluate",
            "implications",
            "in depth",
            "detailed analysis",
            "comprehensive",
            "how does it relate",
            "what are the differences",
        ]

        logger.info(
            f"QueryRouter initialized (LLM classification: {self.use_llm_classification})"
        )

    def _rule_based_classify(self, query: str) -> QueryComplexity:
        """
        Classify query using rule-based heuristics.

        Args:
            query: User query

        Returns:
            QueryComplexity level
        """
        query_lower = query.lower()

        # Check for simple patterns
        if any(keyword in query_lower for keyword in self.simple_keywords):
            return QueryComplexity.SIMPLE

        # Check for complex patterns
        if any(keyword in query_lower for keyword in self.complex_keywords):
            return QueryComplexity.COMPLEX

        # Check query length (simple heuristic)
        word_count = len(query.split())
        if word_count <= 5:
            return QueryComplexity.SIMPLE
        elif word_count > 15:
            return QueryComplexity.COMPLEX

        # Default to medium
        return QueryComplexity.MEDIUM

    def _llm_classify(self, query: str) -> QueryClassification:
        """
        Classify query using LLM for more intelligent routing.

        Args:
            query: User query

        Returns:
            QueryClassification object with detailed classification
        """
        classification_prompt = f"""Classify the following legal query by complexity level.

Query: "{query}"

Classify as:
- SIMPLE: Basic definitions, single-concept explanations (e.g., "What is a contract?")
- MEDIUM: Questions requiring document retrieval but straightforward explanation
- COMPLEX: Multi-faceted questions requiring analysis, comparison, or tool usage

Consider:
1. Number of concepts involved
2. Level of detail/analysis required
3. Whether document search is needed
4. Whether tools (search, comparison) are needed

Provide your classification with reasoning."""

        messages = [
            {
                "role": "system",
                "content": "You are a query classifier for a legal AI assistant. Classify queries accurately by complexity.",
            },
            {"role": "user", "content": classification_prompt},
        ]

        try:
            # Use structured output for reliable classification
            result = self.llm_client.generate_structured(
                messages=messages, response_model=QueryClassification
            )

            logger.info(
                f"LLM classified query as {result.complexity}: {result.reasoning}"
            )
            return result

        except Exception as e:
            logger.error(f"LLM classification failed: {e}. Falling back to rules.")
            # Fallback to rule-based
            complexity = self._rule_based_classify(query)
            return QueryClassification(
                complexity=complexity,
                reasoning="Fallback to rule-based classification",
                requires_rag=(complexity != QueryComplexity.SIMPLE),
                requires_tools=(complexity == QueryComplexity.COMPLEX),
            )

    def classify_query(self, query: str) -> QueryClassification:
        """
        Classify query complexity.

        Args:
            query: User query

        Returns:
            QueryClassification object
        """
        logger.info(f"Classifying query: {query[:50]}...")

        if self.use_llm_classification:
            # Use LLM for intelligent classification
            classification = self._llm_classify(query)
        else:
            # Use rule-based classification
            complexity = self._rule_based_classify(query)
            classification = QueryClassification(
                complexity=complexity,
                reasoning="Rule-based classification",
                requires_rag=(complexity != QueryComplexity.SIMPLE),
                requires_tools=(complexity == QueryComplexity.COMPLEX),
            )

        logger.info(
            f"Query classified as {classification.complexity} (RAG: {classification.requires_rag}, Tools: {classification.requires_tools})"
        )

        return classification

    def route(self, query: str) -> Literal["simple", "medium", "complex"]:
        """
        Route query to appropriate processing path.

        Args:
            query: User query

        Returns:
            Route name: "simple", "medium", or "complex"
        """
        classification = self.classify_query(query)
        return classification.complexity.value

    def should_use_rag(self, query: str) -> bool:
        """
        Determine if query should use RAG retrieval.

        Args:
            query: User query

        Returns:
            True if RAG should be used
        """
        classification = self.classify_query(query)
        return classification.requires_rag

    def should_use_tools(self, query: str) -> bool:
        """
        Determine if query should use tool calling.

        Args:
            query: User query

        Returns:
            True if tools should be used
        """
        classification = self.classify_query(query)
        return classification.requires_tools

    def __repr__(self) -> str:
        return f"QueryRouter(llm_classification={self.use_llm_classification})"


# Example usage and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Test without LLM (rule-based only)
    print("=== Testing Rule-Based Classification ===\n")
    router = QueryRouter(use_llm_classification=False)

    test_queries = [
        "What is a contract?",
        "Define non-disclosure agreement",
        "Explain the implications of copyright infringement in digital media",
        "How does trademark law differ from copyright law?",
        "What are the key elements of a valid contract?",
        "Analyze the liability implications in this scenario",
        "Give me a quick explanation of what liability means",
    ]

    for query in test_queries:
        classification = router.classify_query(query)
        print(f"Query: {query}")
        print(f"  Complexity: {classification.complexity}")
        print(f"  RAG needed: {classification.requires_rag}")
        print(f"  Tools needed: {classification.requires_tools}")
        print(f"  Reasoning: {classification.reasoning}\n")

    # Test with LLM (requires OllamaClient)
    print("\n=== Testing LLM-Based Classification ===\n")
    try:
        from src.llm.ollama_client import OllamaClient

        llm_client = OllamaClient(model="llama3.1:8b")

        if llm_client.test_connection():
            router_llm = QueryRouter(
                llm_client=llm_client, use_llm_classification=True
            )

            # Test a few queries with LLM
            test_query = "Compare and contrast patents and trademarks in detail"
            classification = router_llm.classify_query(test_query)

            print(f"Query: {test_query}")
            print(f"  Complexity: {classification.complexity}")
            print(f"  RAG needed: {classification.requires_rag}")
            print(f"  Tools needed: {classification.requires_tools}")
            print(f"  Reasoning: {classification.reasoning}")
        else:
            print("Ollama not available. Skipping LLM-based classification test.")

    except Exception as e:
        print(f"LLM classification test failed: {e}")
