# Legal Policy Explainer - Local LLM Implementation

## 🎯 Overview

This is a **100% local implementation** of the Legal Policy Explainer using:
- **Ollama** for local LLM inference (Llama 3.1 8B)
- **ChromaDB** for vector storage
- **sentence-transformers** for local embeddings
- **LangGraph** for agent orchestration
- **No API calls** - completely private and offline-capable

## ✨ Features

- ✅ **Zero API costs** - runs entirely on your hardware
- ✅ **Complete privacy** - no data sent to external services
- ✅ **Offline capable** - works without internet connection
- ✅ **Optimized architecture** - unified agent with smart query routing
- ✅ **RAG-powered** - retrieves from your legal document corpus
- ✅ **Safety filters** - built-in guardrails and disclaimers

## 🚀 Quick Start

### 1. Prerequisites

Ensure you have:
- Python 3.10+
- Ollama installed and running
- 16GB+ RAM (for Llama 3.1 8B)
- 20GB+ disk space

### 2. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements_local.txt
```

### 3. Verify Ollama Model

```bash
# Check if llama3.1:8b is available
ollama list

# If not, pull it (this was already done during setup)
ollama pull llama3.1:8b
```

### 4. Add Your Documents

```bash
# Place your legal documents in the data directory
cp your_documents/*.pdf data/documents/
```

Supported formats: `.pdf`, `.txt`, `.docx`

### 5. Ingest Documents

```bash
# Process and ingest documents into ChromaDB
python scripts/ingest_documents.py

# To recreate the vector store (deletes existing data)
python scripts/ingest_documents.py --recreate
```

### 6. Run the Application

```bash
# Interactive mode
python app_local.py

# Single query mode
python app_local.py --query "What is a non-disclosure agreement?"

# Without RAG (LLM knowledge only)
python app_local.py --no-rag
```

## 📁 Project Structure

```
.
├── app_local.py                    # Main application
├── config/
│   └── config_local.yaml           # Local LLM configuration
├── src/
│   ├── llm/
│   │   └── ollama_client.py        # Ollama wrapper with instructor
│   ├── rag/
│   │   ├── embeddings.py           # Local sentence-transformers
│   │   ├── chroma_store.py         # ChromaDB vector store
│   │   ├── rag_retriever.py        # RAG retrieval logic
│   │   ├── document_processor.py   # Document loading & chunking
│   │   └── vector_store.py         # Legacy vector store (reference)
│   ├── agents/
│   │   ├── unified_agent.py        # LangGraph unified agent
│   │   └── query_router.py         # Query complexity router
│   └── prompts/
│       └── system_prompts.py       # All system prompts
├── scripts/
│   ├── ingest_documents.py         # Document ingestion script
│   └── test_setup.py               # Setup verification script
└── data/
    ├── documents/                  # Your legal documents
    └── chroma_db/                  # ChromaDB persistence
```

## 🔧 Configuration

Edit [config/config_local.yaml](config/config_local.yaml) to customize:

### LLM Settings
```yaml
llm:
  model: "llama3.1:8b"    # or mistral:7b, phi3:14b
  temperature: 0.3
  max_tokens: 2000
```

### Embeddings
```yaml
embeddings:
  model: "all-mpnet-base-v2"  # or all-MiniLM-L6-v2 (faster)
  device: "cpu"               # or "cuda" for GPU
```

### RAG Parameters
```yaml
rag:
  top_k: 5                    # Number of documents to retrieve
  similarity_threshold: 0.7   # Minimum similarity score
  chunk_size: 500
  chunk_overlap: 100
```

## 🧪 Testing

### Verify Setup

```bash
python scripts/test_setup.py
```

This tests:
- Ollama connection
- Embeddings generation
- ChromaDB functionality
- Query router
- End-to-end integration

### Test Individual Components

```bash
# Test Ollama client
python src/llm/ollama_client.py

# Test embeddings
python src/rag/embeddings.py

# Test ChromaDB
python src/rag/chroma_store.py

# Test query router
python src/agents/query_router.py
```

## 📊 Architecture

### Query Flow

```
User Query
    ↓
Safety Filter → Checks for inappropriate queries
    ↓
Query Router → Classifies complexity (simple/medium/complex)
    ↓
    ├─→ [Simple Path] Direct LLM response (no RAG)
    ├─→ [Medium Path] LLM + RAG retrieval
    └─→ [Complex Path] LLM + RAG + Tools
    ↓
Add Disclaimer
    ↓
Return Response
```

### Key Components

1. **OllamaClient** - Wraps Ollama with support for:
   - Standard text generation
   - Structured output via instructor
   - Tool/function calling

2. **ChromaVectorStore** - Local vector database:
   - Persistent storage
   - Similarity search with thresholds
   - Metadata filtering

3. **RAGRetriever** - Smart document retrieval:
   - Relevance filtering
   - Context formatting for LLM
   - Max context length management

4. **QueryRouter** - Intelligent routing:
   - Rule-based classification
   - Optional LLM-based classification
   - Determines RAG and tool usage

5. **UnifiedLegalAgent** - LangGraph-based agent:
   - Conditional routing
   - Safety filtering
   - Disclaimer management

## 🎓 Usage Examples

### Interactive Mode

```bash
python app_local.py
```

Example session:
```
Your question: What is a non-disclosure agreement?

Response:
A Non-Disclosure Agreement (NDA) is a legal contract between parties that
outlines confidential information they wish to share with each other for
specific purposes, but want to restrict access to by third parties.

[Based on documents from: business_agreements.pdf]

⚖️ Disclaimer: This is general information only, not legal advice...
```

### Programmatic Usage

```python
from src.llm.ollama_client import OllamaClient
from src.agents.unified_agent import UnifiedLegalAgent

# Initialize
llm_client = OllamaClient(model="llama3.1:8b")
agent = UnifiedLegalAgent(llm_client=llm_client)

# Query
response = agent.query("What is a contract?")
print(response)
```

## ⚙️ Performance Tuning

### For CPU-Only Systems

```yaml
llm:
  model: "mistral:7b"  # Lighter model

embeddings:
  model: "all-MiniLM-L6-v2"  # Faster embeddings
  device: "cpu"

rag:
  top_k: 3  # Fewer retrievals
  chunk_size: 300
```

### For GPU Systems

```yaml
embeddings:
  device: "cuda"
  batch_size: 64

rag:
  top_k: 5
  chunk_size: 500
```

## 🐛 Troubleshooting

### Ollama Not Responding

```bash
# Check if running
ps aux | grep ollama

# Start Ollama
ollama serve &

# Or restart
pkill ollama && ollama serve &
```

### Out of Memory

```bash
# Use smaller model
ollama pull mistral:7b

# Update config
# llm: model: "mistral:7b"
```

### Slow Performance

```bash
# Check if using GPU (if available)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Use quantized model
ollama pull llama3.1:8b-q4  # 4-bit quantized
```

### Import Errors

```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements_local.txt --force-reinstall
```

## 📈 Performance Metrics

**Expected Performance (Llama 3.1 8B on CPU):**
- Response time: 2-5 seconds
- Throughput: ~15 tokens/second
- Memory usage: ~8GB RAM

**With GPU (NVIDIA with 8GB+ VRAM):**
- Response time: 1-2 seconds
- Throughput: ~50-80 tokens/second
- Memory usage: ~6GB VRAM

## 🔒 Privacy & Security

- ✅ No data sent to external APIs
- ✅ All processing happens locally
- ✅ No telemetry or tracking
- ✅ Complete control over data
- ✅ Suitable for sensitive legal documents

## 📚 Next Steps

1. **Add More Documents**: Expand your legal document corpus in `data/documents/`
2. **Fine-Tuning**: See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) Phase 2 for fine-tuning instructions
3. **Customize Prompts**: Edit [src/prompts/system_prompts.py](src/prompts/system_prompts.py)
4. **Add Tools**: Implement custom legal tools in `src/tools/`
5. **Evaluation**: Create test cases and run evaluation metrics

## 🆘 Support

- **Setup Issues**: Check [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md)
- **Implementation Details**: See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- **Architecture**: Review [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Technology Choices**: Read [STACK_COMPARISON.md](STACK_COMPARISON.md)

## 🎉 Success!

You now have a fully functional local LLM-powered Legal Policy Explainer:
- Zero ongoing costs
- Complete privacy
- Production-ready architecture
- Optimized for performance

Enjoy exploring legal documents with AI! 🚀

## 📄 License

[Your License Here]

## 🙏 Acknowledgments

- Ollama for local LLM runtime
- Meta for Llama 3.1
- Sentence Transformers for embeddings
- ChromaDB for vector storage
- LangGraph for agent orchestration
