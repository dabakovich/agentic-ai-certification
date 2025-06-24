import os
from dotenv import load_dotenv
from paths import ENV_FPATH


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
