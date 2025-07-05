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

## Working with Publications

This RAG assistant allows you to insert your own publications into a vector database and then ask questions about them. Here's how to use it:

### Step 1: Prepare Your Publications

Create a JSON file with your publications in the `data/` directory. Each publication must have the following required fields:

```json
[
  {
    "id": "unique-publication-id-1",
    "title": "Your Publication Title",
    "publication_description": "The full content or description of your publication. This is the main text that will be used to answer questions."
  },
  {
    "id": "unique-publication-id-2",
    "title": "Another Publication Title",
    "publication_description": "Another publication's content or description..."
  }
]
```

**Required Fields:**

- `id`: A unique identifier for the publication (string)
- `title`: The title of the publication (string)
- `publication_description`: The main content/description of the publication (string)

### Step 2: Insert Publications into Vector Database

1. Run the application:

   ```bash
   python src/main.py
   ```

2. Choose "Show vector store options" from the main menu

3. Select "Insert new publications"

4. Choose your JSON file from the list of files in the `data/` directory

5. Wait for the publications to be processed and inserted into the vector database

### Step 3: Ask Questions

1. From the main menu, choose "Launch a conversation"

2. Ask questions about your publications - the AI will search through your publications and provide answers based on their content

3. Type 'q' to quit the conversation

The system will automatically find relevant publications based on your questions and provide context-aware answers.

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
