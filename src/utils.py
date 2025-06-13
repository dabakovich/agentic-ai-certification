from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from paths import ENV_FPATH, PUBLICATION_FPATH
from pathlib import Path

GPT_MODEL = "gpt-4o-mini"

# OLLAMA_MODEL = "deepseek-r1:1.5b"
# OLLAMA_MODEL = "llama3.2:1b" # hernia
# OLLAMA_MODEL = "llama3.2:3b" // hernia
OLLAMA_MODEL = "gemma3:4b"


def get_llm(llm: str = "gpt"):
    if llm == "gpt":
        return ChatOpenAI(model_name=GPT_MODEL, temperature=0.0)
    elif llm == "ollama":
        return ChatOllama(model=OLLAMA_MODEL, temperature=0.0)
    else:
        raise ValueError(f"LLM {llm} not supported")
    


def load_env() -> None:
    """Loads environment variables from a .env file and checks for required keys.

    Raises:
        AssertionError: If required keys are missing.
    """
    # Load environment variables from .env file
    load_dotenv(ENV_FPATH, override=True)

    # Check if 'XYZ' has been loaded
    api_key = os.getenv("OPENAI_API_KEY")

    assert api_key, "'api_key' has not been loaded or is not set in the .env file."


def load_publication():
    """Loads the publication markdown file.

    Returns:
        Content of the publication as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there's an error reading the file.
    """
    file_path = Path(PUBLICATION_FPATH)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Publication file not found: {file_path}")

    # Read and return the file content
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except IOError as e:
        raise IOError(f"Error reading publication file: {e}") from e