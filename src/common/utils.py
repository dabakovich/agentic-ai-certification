import os
from dotenv import load_dotenv
from paths import ENV_FPATH
import yaml
from pathlib import Path
from typing import Union
import json


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
            "id": doc["id"],
        }
        for doc in data
    ]

    return publications
