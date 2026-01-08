# Legal & Policy Explainer - LLM Assistant

An intelligent assistant that helps users understand complex legal policies, regulations, and documents through natural language interactions.

## Project Overview

This LLM-powered assistant is designed to:
- Explain legal and policy documents in plain language
- Answer questions about regulations, laws, and institutional policies
- Retrieve relevant information from a corpus of legal documents
- Provide contextual analysis using multi-agent collaboration
- Ensure ethical and safe responses with appropriate disclaimers

## System Architecture

The system implements all required LLM technologies:

1. **Prompt Design**: Specialized system prompts for legal explanation with clear role definitions and safety constraints
2. **RAG (Retrieval-Augmented Generation)**: Vector-based retrieval from legal document corpus
3. **Fine-tuning/PEFT**: LoRA-based adaptation on legal Q&A dataset
4. **Tool/Function Calling**: Legal document search, statute lookup, definition retrieval
5. **Multi-Agent System**:
   - **Researcher Agent**: Retrieves and analyzes relevant legal documents
   - **Explainer Agent**: Translates legal jargon into clear, understandable language
6. **Evaluation**: Test suite with legal queries and expected behaviors
7. **Ethics & Safety**: Built-in disclaimers, refusal rules, and bias mitigation

## Project Structure

```
.
├── src/
│   ├── agents/           # Multi-agent implementation
│   ├── rag/              # Retrieval-augmented generation
│   ├── fine_tuning/      # Model fine-tuning/PEFT
│   ├── tools/            # Function calling tools
│   ├── prompts/          # System prompts and templates
│   └── evaluation/       # Evaluation framework
├── data/
│   ├── documents/        # Legal documents corpus
│   ├── training/         # Fine-tuning datasets
│   └── test_cases/       # Evaluation test cases
├── notebooks/            # Jupyter notebooks for experimentation
├── config/               # Configuration files
├── models/               # Fine-tuned model checkpoints
└── app.py               # Main application entry point
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run the assistant
python app.py

# Run evaluation
python src/evaluation/evaluate.py

# Fine-tune the model
python src/fine_tuning/train.py
```

## Components

### 1. Prompt Design
- System prompts define the assistant's role as a legal explainer
- Structured prompts with examples for consistent responses
- Safety constraints to prevent giving actual legal advice

### 2. RAG System
- Vector database (ChromaDB/FAISS) for document storage
- Semantic search for relevant legal passages
- Context injection into prompts

### 3. Fine-tuning
- LoRA/QLoRA for parameter-efficient fine-tuning
- Training on legal Q&A pairs
- Comparison of base vs. fine-tuned model

### 4. Tools
- Legal document search tool
- Statute/regulation lookup
- Legal term definition retrieval

### 5. Multi-Agent Architecture
- **Researcher Agent**: Finds relevant documents and extracts key information
- **Explainer Agent**: Simplifies legal language and provides clear explanations
- Coordinated workflow for comprehensive responses

## Evaluation

The system is evaluated on:
- **Correctness**: Accuracy of information from legal documents
- **Clarity**: Readability and understandability of explanations
- **Relevance**: Appropriate retrieval of relevant documents
- **Safety**: Proper disclaimers and refusal of inappropriate requests

## Ethics, Safety & Limitations

### Disclaimers
⚠️ **IMPORTANT**: This assistant provides general information about legal policies and regulations for educational purposes only. It does NOT provide legal advice. Always consult a qualified attorney for specific legal matters.

### Safety Rules
- Refuses to provide specific legal advice on pending matters
- Includes disclaimers in all responses
- Avoids speculation about case outcomes
- Directs users to professional legal counsel when appropriate

### Known Limitations
- May hallucinate legal information not present in source documents
- Cannot interpret jurisdiction-specific nuances
- May contain biases from training data
- Not a replacement for professional legal counsel

### Risk Mitigation
- RAG grounds responses in actual documents
- Multi-agent verification reduces hallucinations
- Clear disclaimers prevent misuse
- Evaluation includes adversarial test cases

## Development Roadmap

- [x] Project structure setup
- [ ] Base LLM integration with prompt design
- [ ] RAG system implementation
- [ ] Fine-tuning pipeline
- [ ] Tool integration
- [ ] Multi-agent system
- [ ] Evaluation framework
- [ ] Ethics & safety features
- [ ] Documentation & demo

## Team Members

[Add your team members here]

## License

[Specify license]

## References

- Course Project Document: GenAI Course Project (1).docx
- [Add relevant legal document sources]
- [Add model and library references]
