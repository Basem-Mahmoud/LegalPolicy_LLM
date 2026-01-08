# Implementation Plan - Local LLM Version (v0.3)

## 🎯 Project Goal

Build a **complete, working Legal Policy Explainer** using local open-source LLMs with the optimized unified agent architecture. Fine-tuning will be implemented in Phase 2.

---

## 📋 Implementation Phases

### **Phase 1: Core Application with Local LLM** (Priority)
Build fully functional application with Ollama and open-source models.

### **Phase 2: Fine-tuning & Advanced Features** (Later)
Add fine-tuning capabilities and advanced optimizations.

---

## 🔧 Technology Stack (Local LLM)

```yaml
# Core Infrastructure
LLM Runtime: Ollama
  Primary Model: llama3.1:8b (best balance)
  Backup Model: mistral:7b (faster alternative)

# Framework & Orchestration
Agent Framework: LangGraph
  State Management: StateGraph
  Routing: Conditional edges

# RAG Components
Vector Store: ChromaDB (embedded mode)
Embeddings: sentence-transformers/all-mpnet-base-v2
Text Processing: LangChain (splitters only)

# Function Calling
Structured Output: instructor library
Format: JSON-based tool calls

# UI/Interface
CLI: argparse + rich (colored output)
Optional Web UI: Gradio (Phase 1.5)

# Development
Environment: Python 3.10+
Dependency Management: pip + requirements.txt
Testing: pytest
```

---

## 📦 Phase 1: Core Application (PRIORITY)

**Goal**: Working application with all 7 required components using local LLM.

**Timeline**: Focus on completion, not speed.

### Step 1.1: Environment Setup (Day 1)

**Tasks:**
1. Install Ollama and pull models
2. Set up Python environment
3. Install dependencies
4. Verify installations

**Commands:**
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull models (choose based on your hardware)
ollama pull llama3.1:8b      # Recommended (needs ~8GB RAM)
ollama pull mistral:7b       # Alternative (faster, needs ~6GB RAM)
ollama pull phi3:14b         # Larger model (needs ~14GB RAM)

# Create Python environment
cd /home/omar/GenAI/LegalPolicy_LLM
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install ollama langchain langchain-community langgraph \
            chromadb sentence-transformers instructor \
            python-dotenv pydantic rich pypdf docx2txt \
            pytest pytest-cov
```

**Verification:**
```bash
# Test Ollama
ollama list
ollama run llama3.1:8b "Hello, test"

# Test Python imports
python -c "import langchain, chromadb, sentence_transformers; print('All imports OK')"
```

**Deliverable:** `SETUP_COMPLETE.txt` with installation notes

---

### Step 1.2: Update Configuration (Day 1)

**Tasks:**
1. Create new config for local LLM
2. Update config.yaml with Ollama settings
3. Create .env template (no API keys needed!)

**Files to Create/Update:**
- `config/config_local.yaml` - New config for local setup
- `.env.local` - Local environment template

**New Config Structure:**
```yaml
# config/config_local.yaml
llm:
  provider: "ollama"
  model: "llama3.1:8b"
  base_url: "http://localhost:11434"
  temperature: 0.3
  max_tokens: 2000
  timeout: 120  # Local models can be slower

embeddings:
  provider: "sentence-transformers"
  model: "all-mpnet-base-v2"
  device: "cuda"  # or "cpu" if no GPU

vector_store:
  provider: "chromadb"
  persist_directory: "./data/chroma_db"
  collection_name: "legal_documents"

rag:
  chunk_size: 500
  chunk_overlap: 50
  top_k: 5
  similarity_threshold: 0.7

agent:
  type: "unified"  # Not multi-agent
  enable_query_router: true
  enable_function_calling: true

query_router:
  simple_keywords: ["define", "what is", "meaning of"]
  complex_keywords: ["analyze", "compare", "statute", "reference"]

tools:
  enable_document_search: true
  enable_definition_lookup: true
  enable_statute_lookup: true

safety:
  enable_filter: true
  enable_disclaimer: true

evaluation:
  test_set_path: "./data/test_cases/test_queries.json"

ui:
  interface: "cli"
  enable_colors: true
  streaming: false  # Will add later
```

**Deliverable:** Updated config files

---

### Step 1.3: Build LLM Client Wrapper (Day 2)

**Tasks:**
1. Create unified LLM client for Ollama
2. Add instructor for structured outputs
3. Test basic inference

**File:** `src/llm/ollama_client.py`

**Implementation:**
```python
"""
Ollama LLM client with structured output support.
"""
from typing import Dict, Any, List, Optional
import ollama
import instructor
from openai import OpenAI
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class OllamaClient:
    """Unified client for Ollama LLM with structured outputs."""

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434"):
        """
        Initialize Ollama client.

        Args:
            model: Ollama model name
            base_url: Ollama server URL
        """
        self.model = model
        self.base_url = base_url
        self.ollama_client = ollama.Client(host=base_url)

        # Initialize instructor for structured outputs
        self.instructor_client = instructor.from_openai(
            OpenAI(base_url=f"{base_url}/v1", api_key="ollama"),
            mode=instructor.Mode.JSON
        )

        logger.info(f"Initialized Ollama client with model: {model}")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> str:
        """
        Generate text response.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Enable streaming (not implemented yet)

        Returns:
            Generated text
        """
        try:
            response = self.ollama_client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )
            return response['message']['content']

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise

    def generate_structured(
        self,
        messages: List[Dict[str, str]],
        response_model: BaseModel,
        temperature: float = 0.3
    ) -> BaseModel:
        """
        Generate structured output using instructor.

        Args:
            messages: List of message dicts
            response_model: Pydantic model for structured output
            temperature: Sampling temperature

        Returns:
            Structured response as Pydantic model
        """
        try:
            response = self.instructor_client.chat.completions.create(
                model=self.model,
                messages=messages,
                response_model=response_model,
                temperature=temperature
            )
            return response

        except Exception as e:
            logger.error(f"Error generating structured response: {e}")
            raise

    def test_connection(self) -> bool:
        """Test connection to Ollama server."""
        try:
            response = self.generate(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            logger.info("Ollama connection test successful")
            return True
        except Exception as e:
            logger.error(f"Ollama connection test failed: {e}")
            return False


# Tool call models for structured function calling
class ToolCall(BaseModel):
    """Model for tool/function calls."""
    tool_name: str
    arguments: Dict[str, Any]
    reasoning: str  # Why this tool was chosen


class QueryComplexity(BaseModel):
    """Model for query complexity classification."""
    complexity: str  # "simple", "medium", "complex"
    reasoning: str
    requires_rag: bool
    requires_tools: bool


if __name__ == "__main__":
    # Test the client
    client = OllamaClient()

    # Test basic generation
    response = client.generate([
        {"role": "user", "content": "What is a contract in one sentence?"}
    ])
    print("Basic generation test:")
    print(response)
    print()

    # Test structured output
    structured = client.generate_structured(
        messages=[
            {"role": "user", "content": "Classify this query complexity: 'What is liability?'"}
        ],
        response_model=QueryComplexity
    )
    print("Structured output test:")
    print(structured)
```

**Test File:** `tests/test_ollama_client.py`

**Deliverable:** Working Ollama client with tests

---

### Step 1.4: Build RAG System (Day 3-4)

**Tasks:**
1. Update document processor for local embeddings
2. Update vector store for ChromaDB
3. Implement relevance filtering
4. Test with sample documents

**File 1:** `src/rag/embeddings.py` (NEW)

```python
"""
Local embeddings using sentence-transformers.
"""
from sentence_transformers import SentenceTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)


class LocalEmbeddings:
    """Local embedding model using sentence-transformers."""

    def __init__(self, model_name: str = "all-mpnet-base-v2", device: str = "cpu"):
        """
        Initialize local embeddings.

        Args:
            model_name: sentence-transformers model name
            device: 'cuda' or 'cpu'
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)
        self.model_name = model_name
        logger.info(f"Embedding model loaded on {device}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    @property
    def dimension(self) -> int:
        """Get embedding dimension."""
        return self.model.get_sentence_embedding_dimension()
```

**File 2:** `src/rag/chroma_store.py` (UPDATE vector_store.py)

```python
"""
ChromaDB vector store for local RAG.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional
import logging
from .embeddings import LocalEmbeddings

logger = logging.getLogger(__name__)


class ChromaVectorStore:
    """Vector store using ChromaDB."""

    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "legal_documents",
        embedding_model: str = "all-mpnet-base-v2"
    ):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory to persist database
            collection_name: Name of the collection
            embedding_model: sentence-transformers model name
        """
        self.persist_directory = persist_directory
        self.collection_name = collection_name

        # Initialize embeddings
        self.embeddings = LocalEmbeddings(model_name=embedding_model)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        logger.info(f"ChromaDB initialized: {persist_directory}/{collection_name}")
        logger.info(f"Collection has {self.collection.count()} documents")

    def add_documents(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: Optional[List[str]] = None
    ):
        """
        Add documents to the vector store.

        Args:
            texts: List of document texts
            metadatas: List of metadata dicts
            ids: Optional list of document IDs
        """
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(texts))]

        # Generate embeddings
        embeddings = self.embeddings.embed_documents(texts)

        # Add to collection
        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

        logger.info(f"Added {len(texts)} documents to collection")

    def search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            query: Query text
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score (0-1)
            filter_metadata: Optional metadata filter

        Returns:
            List of result dicts with 'content', 'metadata', 'similarity'
        """
        # Embed query
        query_embedding = self.embeddings.embed_query(query)

        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_metadata
        )

        # Format results
        formatted_results = []
        for i in range(len(results['ids'][0])):
            # Convert distance to similarity (cosine distance: 0=identical, 2=opposite)
            # Similarity = 1 - (distance / 2)
            distance = results['distances'][0][i]
            similarity = 1.0 - (distance / 2.0)

            if similarity >= similarity_threshold:
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': round(similarity, 3)
                })

        logger.info(f"Found {len(formatted_results)} documents above threshold {similarity_threshold}")
        return formatted_results

    def clear(self):
        """Clear all documents from collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(self.collection_name)
        logger.info("Collection cleared")
```

**File 3:** `src/rag/rag_retriever.py` (NEW)

```python
"""
RAG retriever with smart retrieval logic.
"""
from typing import List, Dict, Any, Tuple
from .chroma_store import ChromaVectorStore
from .document_processor import DocumentProcessor
import logging

logger = logging.getLogger(__name__)


class RAGRetriever:
    """RAG retriever with smart retrieval logic."""

    def __init__(
        self,
        vector_store: ChromaVectorStore,
        top_k: int = 5,
        similarity_threshold: float = 0.7
    ):
        """
        Initialize RAG retriever.

        Args:
            vector_store: ChromaVectorStore instance
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
        """
        self.vector_store = vector_store
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        threshold: Optional[float] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for query.

        Args:
            query: User query
            top_k: Override default top_k
            threshold: Override default threshold

        Returns:
            List of retrieved documents
        """
        k = top_k or self.top_k
        thresh = threshold or self.similarity_threshold

        results = self.vector_store.search(
            query=query,
            top_k=k,
            similarity_threshold=thresh
        )

        return results

    def retrieve_and_format(
        self,
        query: str,
        max_context_length: int = 2000
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Retrieve documents and format as context string.

        Args:
            query: User query
            max_context_length: Maximum context length in characters

        Returns:
            Tuple of (documents list, formatted context string)
        """
        documents = self.retrieve(query)

        if not documents:
            return [], ""

        # Format context
        context_parts = ["Retrieved documents:\n"]
        current_length = len(context_parts[0])

        for i, doc in enumerate(documents, 1):
            source = doc['metadata'].get('source', 'Unknown')
            content = doc['content']
            similarity = doc['similarity']

            doc_text = f"\n[Document {i}] (similarity: {similarity})\nSource: {source}\n{content}\n"

            if current_length + len(doc_text) > max_context_length:
                break

            context_parts.append(doc_text)
            current_length += len(doc_text)

        context = "".join(context_parts)
        return documents, context
```

**Deliverable:** Working RAG system with ChromaDB and local embeddings

---

### Step 1.5: Build Query Router (Day 5)

**Tasks:**
1. Implement query complexity classifier
2. Create routing logic
3. Test routing decisions

**File:** `src/agents/query_router.py` (NEW)

```python
"""
Query router for complexity-based routing.
"""
from typing import Dict, Any, Literal
from ..llm.ollama_client import OllamaClient, QueryComplexity
import logging

logger = logging.getLogger(__name__)


class QueryRouter:
    """Routes queries based on complexity."""

    def __init__(
        self,
        llm_client: OllamaClient,
        simple_keywords: List[str] = None,
        complex_keywords: List[str] = None
    ):
        """
        Initialize query router.

        Args:
            llm_client: OllamaClient instance
            simple_keywords: Keywords indicating simple queries
            complex_keywords: Keywords indicating complex queries
        """
        self.llm_client = llm_client
        self.simple_keywords = simple_keywords or [
            "define", "what is", "meaning of", "definition of"
        ]
        self.complex_keywords = complex_keywords or [
            "analyze", "compare", "statute", "reference", "cite",
            "case law", "jurisdiction", "precedent"
        ]

    def classify_query(self, query: str) -> QueryComplexity:
        """
        Classify query complexity using LLM.

        Args:
            query: User query

        Returns:
            QueryComplexity model with classification
        """
        # First try rule-based classification
        query_lower = query.lower()

        # Check for simple patterns
        if any(keyword in query_lower for keyword in self.simple_keywords):
            return QueryComplexity(
                complexity="simple",
                reasoning="Query contains definition keywords",
                requires_rag=False,
                requires_tools=False
            )

        # Check for complex patterns
        if any(keyword in query_lower for keyword in self.complex_keywords):
            return QueryComplexity(
                complexity="complex",
                reasoning="Query contains analysis/reference keywords",
                requires_rag=True,
                requires_tools=True
            )

        # Use LLM for ambiguous cases
        messages = [
            {
                "role": "system",
                "content": """Classify the query complexity:
- simple: Basic definition or explanation (e.g., "What is a contract?")
- medium: Requires document search but not tools (e.g., "Explain liability in contracts")
- complex: Requires document search AND tools (e.g., "Find statute references for GDPR Article 6")

Respond with classification and reasoning."""
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nClassify this query."
            }
        ]

        try:
            classification = self.llm_client.generate_structured(
                messages=messages,
                response_model=QueryComplexity,
                temperature=0.1
            )
            logger.info(f"Query classified as: {classification.complexity}")
            return classification

        except Exception as e:
            logger.error(f"Classification failed, defaulting to medium: {e}")
            return QueryComplexity(
                complexity="medium",
                reasoning="Classification failed, using default",
                requires_rag=True,
                requires_tools=False
            )

    def route(self, query: str) -> Literal["simple", "medium", "complex"]:
        """
        Route query to appropriate handler.

        Args:
            query: User query

        Returns:
            Route name: "simple", "medium", or "complex"
        """
        classification = self.classify_query(query)
        return classification.complexity
```

**Deliverable:** Working query router with classification

---

### Step 1.6: Build Unified Agent with LangGraph (Day 6-7)

This is the core of the optimized architecture!

**File:** `src/agents/unified_agent.py` (NEW)

```python
"""
Unified Legal Assistant Agent using LangGraph.
This replaces the separate Researcher and Explainer agents.
"""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from ..llm.ollama_client import OllamaClient, ToolCall
from ..rag.rag_retriever import RAGRetriever
from ..tools.legal_tools import LegalTools
from ..prompts.system_prompts import (
    UNIFIED_AGENT_PROMPT,
    add_disclaimer
)
from .query_router import QueryRouter
import logging

logger = logging.getLogger(__name__)


# State definition
class AgentState(TypedDict):
    """State for the unified agent graph."""
    query: str
    complexity: str
    safety_check: bool
    safety_message: str
    context: str
    retrieved_docs: list
    tool_calls: list
    tool_results: list
    response: str
    final_response: str


class UnifiedLegalAgent:
    """
    Unified legal assistant agent with query routing.
    Replaces the multi-agent system with a single efficient agent.
    """

    def __init__(
        self,
        llm_client: OllamaClient,
        rag_retriever: RAGRetriever,
        legal_tools: LegalTools,
        enable_query_router: bool = True
    ):
        """
        Initialize unified agent.

        Args:
            llm_client: OllamaClient instance
            rag_retriever: RAG retriever
            legal_tools: Legal tools instance
            enable_query_router: Enable smart routing
        """
        self.llm_client = llm_client
        self.rag_retriever = rag_retriever
        self.legal_tools = legal_tools
        self.enable_router = enable_query_router

        if enable_query_router:
            self.query_router = QueryRouter(llm_client)

        # Build the graph
        self.app = self._build_graph()

        logger.info("Unified Legal Agent initialized")

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""

        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("safety_filter", self._safety_filter)
        workflow.add_node("route_query", self._route_query)
        workflow.add_node("simple_path", self._simple_path)
        workflow.add_node("medium_path", self._medium_path)
        workflow.add_node("complex_path", self._complex_path)
        workflow.add_node("add_disclaimer", self._add_disclaimer)

        # Set entry point
        workflow.set_entry_point("safety_filter")

        # Add edges
        workflow.add_conditional_edges(
            "safety_filter",
            self._check_safety,
            {
                "safe": "route_query",
                "unsafe": "add_disclaimer"
            }
        )

        workflow.add_conditional_edges(
            "route_query",
            self._get_route,
            {
                "simple": "simple_path",
                "medium": "medium_path",
                "complex": "complex_path"
            }
        )

        workflow.add_edge("simple_path", "add_disclaimer")
        workflow.add_edge("medium_path", "add_disclaimer")
        workflow.add_edge("complex_path", "add_disclaimer")
        workflow.add_edge("add_disclaimer", END)

        return workflow.compile()

    # Node functions
    def _safety_filter(self, state: AgentState) -> AgentState:
        """Check if query is safe."""
        query = state["query"]

        # Simple safety check (can be enhanced)
        unsafe_patterns = [
            "how to break the law",
            "avoid taxes",
            "evade",
            "illegal"
        ]

        query_lower = query.lower()
        is_safe = not any(pattern in query_lower for pattern in unsafe_patterns)

        state["safety_check"] = is_safe
        if not is_safe:
            state["safety_message"] = "I cannot provide guidance on circumventing or violating laws."

        logger.info(f"Safety check: {'PASS' if is_safe else 'FAIL'}")
        return state

    def _route_query(self, state: AgentState) -> AgentState:
        """Classify query complexity."""
        if self.enable_router:
            complexity = self.query_router.route(state["query"])
        else:
            complexity = "medium"  # Default to medium

        state["complexity"] = complexity
        logger.info(f"Query routed to: {complexity}")
        return state

    def _simple_path(self, state: AgentState) -> AgentState:
        """Handle simple definition queries."""
        query = state["query"]

        # Try to extract term
        # Simple pattern matching: "what is X", "define X", etc.
        query_lower = query.lower()
        for pattern in ["what is ", "define ", "meaning of "]:
            if pattern in query_lower:
                term = query_lower.split(pattern)[-1].strip("?.,!").strip()
                break
        else:
            term = query

        # Lookup definition
        definition = self.legal_tools.get_legal_definition(term)

        if "don't have a specific definition" not in definition:
            # Found in dictionary
            state["response"] = definition
        else:
            # Not in dictionary, use LLM with simple prompt
            messages = [
                {
                    "role": "system",
                    "content": "You are a legal explainer. Provide a clear, concise definition."
                },
                {
                    "role": "user",
                    "content": f"Define '{term}' in legal terms."
                }
            ]
            response = self.llm_client.generate(messages, temperature=0.2)
            state["response"] = response

        logger.info("Simple path completed")
        return state

    def _medium_path(self, state: AgentState) -> AgentState:
        """Handle medium complexity queries with RAG."""
        query = state["query"]

        # Retrieve documents
        docs, context = self.rag_retriever.retrieve_and_format(query)
        state["retrieved_docs"] = docs
        state["context"] = context

        # Generate response with context
        messages = [
            {
                "role": "system",
                "content": UNIFIED_AGENT_PROMPT
            },
            {
                "role": "user",
                "content": f"""Context from legal documents:
{context}

User question: {query}

Provide a clear, accurate explanation based on the context provided."""
            }
        ]

        response = self.llm_client.generate(messages, temperature=0.3)
        state["response"] = response

        logger.info("Medium path completed")
        return state

    def _complex_path(self, state: AgentState) -> AgentState:
        """Handle complex queries with RAG + tools."""
        query = state["query"]

        # First retrieve documents
        docs, context = self.rag_retriever.retrieve_and_format(query)
        state["retrieved_docs"] = docs
        state["context"] = context

        # Check if tools are needed
        tool_decision_messages = [
            {
                "role": "system",
                "content": """You are a legal assistant. Determine if any tools are needed.
Available tools:
- search_legal_documents: Search document database
- get_legal_definition: Get term definition
- check_statute_reference: Look up statute

Respond with tool calls if needed."""
            },
            {
                "role": "user",
                "content": f"Query: {query}\n\nWhat tools should I use?"
            }
        ]

        try:
            tool_call = self.llm_client.generate_structured(
                messages=tool_decision_messages,
                response_model=ToolCall,
                temperature=0.2
            )

            # Execute tool
            tool_result = self.legal_tools.execute_tool(
                tool_call.tool_name,
                tool_call.arguments
            )

            state["tool_calls"] = [tool_call.dict()]
            state["tool_results"] = [tool_result]

            # Generate final response with all context
            final_messages = [
                {
                    "role": "system",
                    "content": UNIFIED_AGENT_PROMPT
                },
                {
                    "role": "user",
                    "content": f"""Retrieved documents:
{context}

Tool used: {tool_call.tool_name}
Tool result: {tool_result}

User question: {query}

Provide a comprehensive answer using all available information."""
                }
            ]

        except Exception as e:
            logger.warning(f"Tool execution failed: {e}, using RAG only")
            # Fallback to RAG only
            final_messages = [
                {
                    "role": "system",
                    "content": UNIFIED_AGENT_PROMPT
                },
                {
                    "role": "user",
                    "content": f"""Context: {context}\n\nQuestion: {query}"""
                }
            ]

        response = self.llm_client.generate(final_messages, temperature=0.3)
        state["response"] = response

        logger.info("Complex path completed")
        return state

    def _add_disclaimer(self, state: AgentState) -> AgentState:
        """Add legal disclaimer to response."""
        if not state["safety_check"]:
            state["final_response"] = state["safety_message"]
        else:
            state["final_response"] = add_disclaimer(state["response"])

        return state

    # Conditional edge functions
    def _check_safety(self, state: AgentState) -> Literal["safe", "unsafe"]:
        """Route based on safety check."""
        return "safe" if state["safety_check"] else "unsafe"

    def _get_route(self, state: AgentState) -> Literal["simple", "medium", "complex"]:
        """Get routing decision."""
        return state["complexity"]

    # Public API
    def query(self, user_query: str) -> Dict[str, Any]:
        """
        Process a user query.

        Args:
            user_query: User's question

        Returns:
            Dict with response and metadata
        """
        logger.info(f"Processing query: {user_query[:50]}...")

        # Initialize state
        initial_state = {
            "query": user_query,
            "complexity": "",
            "safety_check": True,
            "safety_message": "",
            "context": "",
            "retrieved_docs": [],
            "tool_calls": [],
            "tool_results": [],
            "response": "",
            "final_response": ""
        }

        # Run the graph
        final_state = self.app.invoke(initial_state)

        # Return results
        return {
            "query": user_query,
            "response": final_state["final_response"],
            "complexity": final_state["complexity"],
            "retrieved_docs": final_state["retrieved_docs"],
            "tool_calls": final_state["tool_calls"]
        }
```

**Deliverable:** Complete unified agent with LangGraph routing

---

**(Character limit reached - continuing in next section)**

This implementation plan continues with:
- Step 1.7: Update prompts for unified agent
- Step 1.8: Build CLI interface
- Step 1.9: Testing and validation
- Step 1.10: Documentation updates
- Phase 2: Fine-tuning implementation

Would you like me to continue with the rest of the implementation plan?
