# Simple RAG Assistant

A Python-based RAG (Retrieval-Augmented Generation) assistant project.

## Setup

1. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
.\venv\Scripts\activate  # On Windows
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## LLM Setup

### Option 1: OpenAI API

1. Get your API key from [OpenAI](https://platform.openai.com/api-keys)
2. Create a `.env` file in the project root and add your API key:

```
OPENAI_API_KEY=your-api-key-here
```

### Option 2: Local LLM with Ollama

1. Install Ollama from [ollama.ai](https://ollama.ai)
2. Run the model:

```bash
ollama run gemma3
```

## Project Structure

- `src/` - Source code directory
- `data/` - Data files
- `requirements.txt` - Project dependencies
