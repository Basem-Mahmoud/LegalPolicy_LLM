# Quick Start Guide - Local LLM Setup

## Step-by-Step Setup

### 1. Activate Virtual Environment

```bash
# Navigate to project directory
cd /home/omar/GenAI/LegalPolicy_LLM

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your prompt
```

### 2. Verify Ollama is Running

```bash
# Check Ollama status
ollama list

# If not running, start it
ollama serve &

# Verify llama3.1:8b is available
ollama list | grep llama3.1
```

### 3. Test the Setup

```bash
# Run verification tests
python scripts/test_setup.py

# This will test:
# - Ollama connection
# - Embeddings generation
# - ChromaDB functionality
# - Query router
# - End-to-end integration
```

### 4. Add Sample Documents (Optional)

```bash
# Create some sample legal documents for testing
cat > data/documents/sample_contract.txt << 'EOF'
CONTRACT DEFINITION

A contract is a legally binding agreement between two or more parties that creates
mutual obligations enforceable by law. The basic elements of a contract include:

1. Offer: One party makes a clear proposal
2. Acceptance: The other party agrees to the terms
3. Consideration: Something of value is exchanged
4. Legal capacity: Parties are legally able to enter the contract
5. Legal purpose: The contract's purpose is lawful

Contracts can be written, oral, or implied by conduct. However, certain types
of contracts must be in writing to be enforceable under the Statute of Frauds.
EOF

cat > data/documents/nda_info.txt << 'EOF'
NON-DISCLOSURE AGREEMENT (NDA)

A Non-Disclosure Agreement (NDA) is a legal contract between parties that
outlines confidential information that the parties wish to share with each
other for specific purposes, but want to restrict access to by third parties.

Key Components:
- Definition of confidential information
- Obligations of the receiving party
- Exceptions to confidentiality
- Duration of the confidentiality obligation
- Consequences of breach

Types of NDAs:
- Unilateral: One party discloses information
- Bilateral/Mutual: Both parties share confidential information
- Multilateral: Three or more parties are involved

Common Uses:
- Business partnerships
- Employment relationships
- Investor discussions
- Vendor/supplier relationships
EOF

echo "✓ Sample documents created!"
```

### 5. Ingest Documents

```bash
# Process and ingest documents into ChromaDB
python scripts/ingest_documents.py

# Expected output:
# - Loading documents...
# - Generating embeddings...
# - Adding to ChromaDB...
# - ✓ Documents successfully ingested!
```

### 6. Run the Application

```bash
# Interactive mode (recommended for first use)
python app_local.py

# Try these example queries:
# - What is a contract?
# - Explain non-disclosure agreements
# - What are the key elements of a valid contract?
```

## Alternative: Single Query Mode

```bash
# Test with a single query
python app_local.py --query "What is a non-disclosure agreement?"

# Without RAG (LLM knowledge only)
python app_local.py --no-rag --query "Define contract"
```

## Common Commands

### Deactivate Virtual Environment
```bash
deactivate
```

### Reactivate Later
```bash
cd /home/omar/GenAI/LegalPolicy_LLM
source venv/bin/activate
```

### Update Dependencies
```bash
pip install -r requirements_local.txt --upgrade
```

### Reingest Documents (Fresh Start)
```bash
# This deletes existing ChromaDB and reingests everything
python scripts/ingest_documents.py --recreate
```

### Check Ollama Models
```bash
# List installed models
ollama list

# Pull a different model
ollama pull mistral:7b

# Update config to use different model
nano config/config_local.yaml
# Change: llm -> model: "mistral:7b"
```

## Troubleshooting

### Issue: Cannot connect to Ollama

**Solution:**
```bash
# Check if Ollama is running
ps aux | grep ollama

# If not running, start it
ollama serve &

# Wait a few seconds, then test
ollama list
```

### Issue: Import errors

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements_local.txt --force-reinstall
```

### Issue: No documents found

**Solution:**
```bash
# Check documents directory
ls -la data/documents/

# Add documents
cp /path/to/your/documents/*.pdf data/documents/

# Reingest
python scripts/ingest_documents.py --recreate
```

### Issue: Slow responses

**Solutions:**
```bash
# Option 1: Use smaller model
ollama pull mistral:7b
# Update config_local.yaml: model: "mistral:7b"

# Option 2: Use quantized version
ollama pull llama3.1:8b-q4

# Option 3: Reduce max_tokens in config
nano config/config_local.yaml
# Change: max_tokens: 1000
```

## Expected Performance

**First Run:**
- Document ingestion: 1-5 minutes (depending on document count)
- First query: 5-10 seconds (model loading)

**Subsequent Runs:**
- Query response: 2-5 seconds (CPU)
- Query response: 1-2 seconds (GPU)

## Next Steps

Once everything is working:

1. **Add Your Documents**: Place your legal documents in `data/documents/`
2. **Customize Configuration**: Edit `config/config_local.yaml` for your needs
3. **Try Different Queries**: Test simple, medium, and complex queries
4. **Explore the Code**: Check out the implementation in `src/`
5. **Read Documentation**: See `README_LOCAL.md` for detailed information

## Quick Reference

| Command | Purpose |
|---------|---------|
| `source venv/bin/activate` | Activate virtual environment |
| `ollama serve &` | Start Ollama server |
| `python scripts/test_setup.py` | Verify setup |
| `python scripts/ingest_documents.py` | Ingest documents |
| `python app_local.py` | Run interactive mode |
| `python app_local.py --query "..."` | Single query |
| `deactivate` | Deactivate virtual environment |

## Success Indicators

✅ Virtual environment activated (you see `(venv)` in prompt)
✅ Ollama running (`ollama list` works)
✅ Model available (llama3.1:8b in `ollama list`)
✅ Tests passing (`python scripts/test_setup.py` shows all green)
✅ Documents ingested (ChromaDB has documents)
✅ Application responds to queries

**You're ready to go!** 🚀
