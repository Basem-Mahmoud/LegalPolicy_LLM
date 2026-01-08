# Local LLM Setup Guide

Complete guide for setting up the Legal Policy Explainer with local open-source LLMs.

## 📋 Prerequisites

### Hardware Requirements

**Minimum (for llama3.1:8b or mistral:7b):**
- CPU: 4+ cores
- RAM: 16GB
- Storage: 20GB free space
- No GPU required (but recommended)

**Recommended (for better performance):**
- CPU: 8+ cores
- RAM: 32GB
- Storage: 50GB SSD
- GPU: NVIDIA GPU with 8GB+ VRAM (for faster inference)

**Optimal (for llama3.1:70b or fine-tuning):**
- CPU: 16+ cores
- RAM: 64GB+
- Storage: 100GB SSD
- GPU: NVIDIA GPU with 24GB+ VRAM

### Software Requirements
- Ubuntu 20.04+ / macOS 12+ / Windows 11 with WSL2
- Python 3.10 or higher
- CUDA 11.8+ (if using GPU)
- Git

---

## 🚀 Installation Steps

### Step 1: Install Ollama

**Ubuntu/Debian:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**macOS:**
```bash
# Download from https://ollama.com/download
# Or use Homebrew
brew install ollama
```

**Windows (WSL2):**
```bash
# Run in WSL2 terminal
curl -fsSL https://ollama.com/install.sh | sh
```

**Verify Installation:**
```bash
ollama --version
ollama serve  # Start the server
```

### Step 2: Pull LLM Models

**Choose based on your hardware:**

```bash
# For 16GB RAM (Recommended starting point)
ollama pull llama3.1:8b

# For 8-12GB RAM (Faster, lighter)
ollama pull mistral:7b

# For 32GB+ RAM (Better quality)
ollama pull llama3.1:70b

# Alternative: Microsoft Phi-3 (good balance)
ollama pull phi3:14b
```

**Test the model:**
```bash
ollama run llama3.1:8b "What is a contract in one sentence?"
```

### Step 3: Set Up Python Environment

```bash
# Navigate to project directory
cd /home/omar/GenAI/LegalPolicy_LLM

# Create virtual environment
python3.10 -m venv venv

# Activate it
source venv/bin/activate  # Linux/macOS
# OR
.\venv\Scripts\activate  # Windows

# Upgrade pip
pip install --upgrade pip setuptools wheel
```

### Step 4: Install Python Dependencies

```bash
# Install from local requirements
pip install -r requirements_local.txt

# If you have a GPU with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installations
python -c "import ollama, langchain, chromadb, sentence_transformers; print('✓ All imports successful')"
```

### Step 5: Download Embedding Model

```bash
# This will download on first use, but you can pre-download:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-mpnet-base-v2')"
```

### Step 6: Set Up Project Structure

```bash
# Create necessary directories
mkdir -p data/documents
mkdir -p data/chroma_db
mkdir -p data/test_cases
mkdir -p models
mkdir -p logs

# Copy sample legal documents (if available)
cp path/to/your/legal/docs/*.pdf data/documents/
```

### Step 7: Configure Application

```bash
# No .env file needed for local setup! (no API keys)
# But you can create one for custom settings

cat > .env.local << 'EOF'
# Local LLM Configuration
OLLAMA_MODEL=llama3.1:8b
OLLAMA_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=all-mpnet-base-v2
DEVICE=cuda  # or 'cpu' if no GPU
LOG_LEVEL=INFO
EOF
```

### Step 8: Initialize Vector Database

```bash
# Run the document ingestion script (will be created in implementation)
python scripts/ingest_documents.py --documents data/documents/

# This will:
# 1. Load all documents from data/documents/
# 2. Chunk them appropriately
# 3. Generate embeddings
# 4. Store in ChromaDB
```

### Step 9: Test the Installation

```bash
# Test Ollama connection
python -c "
from src.llm.ollama_client import OllamaClient
client = OllamaClient()
if client.test_connection():
    print('✓ Ollama connection successful')
"

# Test RAG system
python -c "
from src.rag.chroma_store import ChromaVectorStore
store = ChromaVectorStore()
print(f'✓ ChromaDB initialized with {store.collection.count()} documents')
"

# Run full system test
python tests/test_local_setup.py
```

---

## 🧪 Quick Verification Tests

### Test 1: LLM Generation
```bash
python << 'EOF'
from src.llm.ollama_client import OllamaClient

client = OllamaClient(model="llama3.1:8b")
response = client.generate([
    {"role": "user", "content": "What is a contract in one sentence?"}
])
print("LLM Response:", response)
EOF
```

### Test 2: Embeddings
```bash
python << 'EOF'
from src.rag.embeddings import LocalEmbeddings

embedder = LocalEmbeddings()
embedding = embedder.embed_query("What is liability?")
print(f"✓ Embedding dimension: {len(embedding)}")
EOF
```

### Test 3: Vector Search
```bash
python << 'EOF'
from src.rag.chroma_store import ChromaVectorStore

store = ChromaVectorStore()
results = store.search("contract definition", top_k=3)
print(f"✓ Found {len(results)} relevant documents")
EOF
```

### Test 4: Full Pipeline
```bash
python app.py --query "What is a non-disclosure agreement?" --mode local
```

---

## 🔧 Troubleshooting

### Issue: Ollama not responding

**Solution:**
```bash
# Check if Ollama is running
ps aux | grep ollama

# If not, start it
ollama serve &

# Or restart it
pkill ollama
ollama serve &
```

### Issue: CUDA out of memory (GPU)

**Solutions:**
```bash
# 1. Use smaller model
ollama pull mistral:7b

# 2. Use CPU instead
export DEVICE=cpu

# 3. Reduce batch size in config
# Edit config/config_local.yaml:
# embedding_batch_size: 16  # Reduce from 32
```

### Issue: Slow inference

**Solutions:**
```bash
# 1. Check if using GPU
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# 2. Use quantized model
ollama pull llama3.1:8b-q4  # 4-bit quantized version

# 3. Reduce max_tokens in config
# Edit config/config_local.yaml:
# max_tokens: 1000  # Instead of 2000
```

### Issue: Import errors

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements_local.txt --force-reinstall

# Check Python version
python --version  # Should be 3.10+
```

### Issue: ChromaDB permission errors

**Solution:**
```bash
# Fix permissions
chmod -R 755 data/chroma_db

# Or recreate the database
rm -rf data/chroma_db
python scripts/ingest_documents.py
```

### Issue: Sentence-transformers download fails

**Solution:**
```bash
# Manually download model
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')
print('Model downloaded successfully')
"

# Or use alternate model
# Edit config_local.yaml:
# embedding_model: "all-MiniLM-L6-v2"  # Smaller, faster
```

---

## 🎯 Performance Tuning

### For CPU-only Systems

```yaml
# config/config_local.yaml
llm:
  model: "mistral:7b"  # Lighter model
  max_tokens: 1000      # Reduce max length

embeddings:
  model: "all-MiniLM-L6-v2"  # Faster embeddings
  device: "cpu"

rag:
  top_k: 3              # Fewer retrievals
  chunk_size: 300       # Smaller chunks
```

### For GPU Systems

```yaml
# config/config_local.yaml
llm:
  model: "llama3.1:8b"
  max_tokens: 2000

embeddings:
  model: "all-mpnet-base-v2"
  device: "cuda"
  batch_size: 64        # Larger batches

rag:
  top_k: 5
  chunk_size: 500
```

### For High-Memory Systems (64GB+ RAM)

```yaml
# config/config_local.yaml
llm:
  model: "llama3.1:70b"  # Larger model
  max_tokens: 3000

embeddings:
  model: "BAAI/bge-large-en-v1.5"  # Best quality
  device: "cuda"
  batch_size: 128

rag:
  top_k: 10             # More context
  chunk_size: 800
```

---

## 📊 Model Comparison

| Model | Size | RAM Required | Speed | Quality | Best For |
|-------|------|-------------|-------|---------|----------|
| mistral:7b | 4.1GB | 8GB | Fast | Good | CPU systems |
| llama3.1:8b | 4.7GB | 16GB | Medium | Very Good | Balanced (recommended) |
| phi3:14b | 7.9GB | 16GB | Medium | Very Good | Balanced alternative |
| llama3.1:70b | 40GB | 64GB | Slow | Excellent | High-end systems |

**Recommendation for your use case:**
- **Development/Testing**: `mistral:7b` or `llama3.1:8b`
- **Production/Demo**: `llama3.1:8b` or `phi3:14b`
- **Final Presentation**: `llama3.1:70b` (if hardware allows)

---

## 🎓 Next Steps

After successful setup:

1. **Ingest Documents**: Add your legal documents to `data/documents/`
2. **Test Queries**: Run sample queries to verify everything works
3. **Evaluate**: Run evaluation suite to benchmark performance
4. **Optimize**: Tune configuration based on your hardware
5. **Document**: Update docs with your specific setup notes

**Ready to start?** Run:
```bash
python app.py --mode local
```

---

## 🆘 Getting Help

- Check logs: `tail -f logs/app.log`
- Run tests: `pytest tests/ -v`
- Debug mode: `python app.py --debug --verbose`
- Ollama logs: `journalctl -u ollama -f`

**Common Commands:**
```bash
# Check Ollama models
ollama list

# Check Ollama status
ollama ps

# Remove a model (to free space)
ollama rm llama3.1:8b

# Update a model
ollama pull llama3.1:8b

# Check system resources
nvidia-smi  # GPU
free -h     # RAM
df -h       # Disk
```
