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

## Usage

To run the RAG assistant, run the main module from the project's root directory:

```bash
python src/main.py
```

## Project Structure

The project follows a modular structure to separate concerns and facilitate future development.

- `data/`: Stores raw data files for the RAG model.
- `requirements.txt`: Lists all Python dependencies for the project.
- `src/rag_assistant/`: The main application package, organized as follows:
  - `main.py`: The entry point of the application.
  - `paths.py`: Defines all relevant paths used throughout the project.
  - `common/`: Contains shared utilities (`utils.py`) and constants (`constants.py`).
  - `config/`: Holds configuration files, such as `prompt.yaml` for prompt templates and `models.yaml` for LLM definitions.
  - `core/`: Includes the core application logic, like the main conversation loop (`conversation.py`).
  - `llm/`: Manages interactions with language models, including the client (`client.py`) and prompt building (`prompt_builder.py`).
  - `storage/`: (Future) For handling session management and data persistence.
  - `vector_store/`: (Future) For managing the vector database, including data ingestion and retrieval.
