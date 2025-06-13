from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
from paths import ENV_FPATH, PUBLICATION_PATH, PUBLICATION_MINI_PATH
from pathlib import Path
from langchain.chat_models.base import BaseChatModel
from typing import Union, Optional
import yaml
import json

GPT_MODEL = "gpt-4o-mini"

# OLLAMA_MODEL = "deepseek-r1:1.5b"
OLLAMA_MODEL = "llama3.1:8b"
# OLLAMA_MODEL = "llama3.2:1b" # hernia
# OLLAMA_MODEL = "llama3.2:3b" # hernia
# OLLAMA_MODEL = "gemma3:4b"


def get_llm(llm: str = "gpt"):
    if llm == "gpt":
        return ChatOpenAI(model_name=GPT_MODEL, temperature=0.0)
    elif llm == "ollama":
        return ChatOllama(model=OLLAMA_MODEL, temperature=0.0)
    else:
        raise ValueError(f"LLM {llm} not supported")
    

def get_response_with_streaming_to_terminal(llm: BaseChatModel, prompt: any):
    content = ""
    for event in llm.stream(prompt):
        content += event.content
        print(event.content, end="", flush=True)
    
    # Create a response object with a content attribute
    class Response:
        def __init__(self, content):
            self.content = content
    
    return Response(content)


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
    # file_path = Path(PUBLICATION_PATH)
    file_path = Path(PUBLICATION_MINI_PATH)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"Publication file not found: {file_path}")

    # Read and return the file content
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except IOError as e:
        raise IOError(f"Error reading publication file: {e}") from e

# Returns list of publications with title and description
def load_publications(json_path):
    """Load .json file and return as list of publications strings"""
    
    # Load the .json file with array of objects
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    # Print the number of publications
    print(f"\nTotal publications loaded: {len(data)}")

    # Extract publication description as strings and return
    publications = [
        {
            "title": doc["title"],
            "description": doc["publication_description"],
            "id": doc["id"]
        } for doc in data
    ]
    
    return publications


def save_text_to_file(
    text: str, filepath: Union[str, Path], header: Optional[str] = None
) -> None:
    """Saves text content to a file, optionally with a header.

    Args:
        text: The content to write.
        filepath: Destination path for the file.
        header: Optional header text to include at the top.

    Raises:
        IOError: If the file cannot be written.
    """
    try:
        filepath = Path(filepath)

        # Create directory if it doesn't exist
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8") as f:
            if header:
                f.write(f"# {header}\n")
                f.write("# " + "=" * 60 + "\n\n")
            f.write(text)

    except IOError as e:
        raise IOError(f"Error writing to file {filepath}: {e}") from e


def load_yaml_config(file_path: Union[str, Path]) -> dict:
    """Loads a YAML configuration file.

    Args:
        file_path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        yaml.YAMLError: If there's an error parsing YAML.
        IOError: If there's an error reading the file.
    """
    file_path = Path(file_path)

    # Check if file exists
    if not file_path.exists():
        raise FileNotFoundError(f"YAML config file not found: {file_path}")

    # Read and parse the YAML file
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e
    except IOError as e:
        raise IOError(f"Error reading YAML file: {e}") from e