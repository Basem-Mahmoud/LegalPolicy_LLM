# Quick Start Guide

Get up and running with the Legal Policy Explainer in minutes.

## Prerequisites

- Python 3.8 or higher
- API key for OpenAI, Anthropic, or access to HuggingFace models
- 8GB+ RAM recommended
- GPU recommended for fine-tuning (optional)

## Installation

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd LegalPolicy_LLM
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your API keys
```

Example `.env` file:
```
OPENAI_API_KEY=sk-your-key-here
# OR
ANTHROPIC_API_KEY=your-key-here
```

### 5. (Optional) Add your legal documents
Place your legal documents (PDF, TXT, or DOCX) in:
```
data/documents/
```

Sample documents are already included for demonstration.

## Quick Test

### Run the interactive assistant
```bash
python app.py
```

### Run a single query
```bash
python app.py --query "What is a non-disclosure agreement?"
```

### Run evaluation
```bash
python src/evaluation/evaluate.py
```

## Configuration

Edit `config/config.yaml` to customize:
- Model selection (OpenAI, Anthropic, HuggingFace)
- RAG parameters (chunk size, top-k retrieval)
- Multi-agent settings
- Safety and ethics rules

## Next Steps

1. **Add your documents**: Place legal documents in `data/documents/`
2. **Customize prompts**: Edit `src/prompts/system_prompts.py`
3. **Fine-tune model**: Run `python src/fine_tuning/train.py` (requires GPU)
4. **Evaluate**: Run evaluation and adjust based on results
5. **Explore demo**: Open `notebooks/demo.ipynb` in Jupyter

## Common Issues

### API Key Not Found
Make sure your `.env` file is in the project root and contains valid API keys.

### CUDA Out of Memory (Fine-tuning)
Reduce batch size in `config/config.yaml`:
```yaml
fine_tuning:
  batch_size: 2  # Reduce from 4
```

### No Documents Found
Ensure documents are in `data/documents/` directory with supported formats (.pdf, .txt, .docx).

### Import Errors
Make sure you're in the project root directory and virtual environment is activated.

## Architecture Overview

```
User Query
    ↓
Safety Filter → Block inappropriate queries
    ↓
Multi-Agent Orchestrator
    ↓
Researcher Agent → RAG Retrieval → Vector Store
    ↓
Explainer Agent → Generate Response
    ↓
Add Disclaimer
    ↓
Return to User
```

## Components

1. **LLM Client** (`src/llm_client.py`): Unified interface for different LLM providers
2. **RAG System** (`src/rag/`): Document processing and retrieval
3. **Multi-Agent** (`src/agents/`): Researcher and Explainer collaboration
4. **Tools** (`src/tools/`): Function calling capabilities
5. **Prompts** (`src/prompts/`): Carefully designed system prompts
6. **Evaluation** (`src/evaluation/`): Assessment framework
7. **Fine-tuning** (`src/fine_tuning/`): LoRA/PEFT training

## Usage Examples

### Basic Query
```python
from app import LegalPolicyExplainer

assistant = LegalPolicyExplainer()
response = assistant.query("What is copyright?")
print(response)
```

### Without RAG
```python
assistant = LegalPolicyExplainer(use_rag=False)
```

### Single Agent Mode
```python
assistant = LegalPolicyExplainer(use_multi_agent=False)
```

### Evaluate System
```python
from src.evaluation import LegalExplainerEvaluator

evaluator = LegalExplainerEvaluator()
results = evaluator.evaluate_system(assistant.query)
evaluator.print_summary(results)
```

## Project Structure

```
.
├── src/
│   ├── agents/          # Multi-agent system
│   ├── rag/             # RAG implementation
│   ├── tools/           # Function calling tools
│   ├── prompts/         # System prompts
│   ├── fine_tuning/     # PEFT training
│   ├── evaluation/      # Evaluation framework
│   └── llm_client.py    # LLM interface
├── data/
│   ├── documents/       # Legal documents corpus
│   ├── training/        # Fine-tuning data
│   └── test_cases/      # Evaluation cases
├── notebooks/           # Jupyter demos
├── config/              # Configuration files
├── app.py              # Main application
└── README.md
```

## Support

For issues or questions:
1. Check the [README](README.md)
2. Review the demo notebook: `notebooks/demo.ipynb`
3. Check configuration: `config/config.yaml`
4. Review logs for error messages

## What's Next?

1. **Customize for your domain**: Add domain-specific documents and training data
2. **Fine-tune**: Train the model on your specific legal Q&A pairs
3. **Evaluate**: Run comprehensive evaluation and iterate
4. **Deploy**: Set up as a web service or integrate into your application

Enjoy using the Legal Policy Explainer!
