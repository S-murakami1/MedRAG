# MedRAG
MedRAG is a Retrieval-Augmented Generation (RAG) system that generates comprehensive answers to medical questions by leveraging ArXiv, PubMed, and web search. It can also display references.

## Environment Setup

### Installing Dependencies

```bash
# Move to the project directory
cd MedRAG

# Set up the Poetry environment
poetry install
```

### Setting Environment Variables (for Windows)

```bash
# OpenAI API configuration
$env:OPENAI_API_KEY="your_openai_api_key_here"

# Tavily API configuration (for web search)
$env:TAVILY_API_KEY="your_tavily_api_key_here"
```
## Run

```bash
# Run the Python script within the Poetry environment
poetry run python medrag/eng.py -q "Enter your question here"
