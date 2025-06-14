import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENV_FPATH = os.path.join(ROOT_DIR, ".env")

SRC_DIR = os.path.join(ROOT_DIR, "src")

CODE_DIR = os.path.join(ROOT_DIR, "src")
PROMPT_CONFIG_FPATH = os.path.join(CODE_DIR, "config", "prompt_config.yaml")

OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")


DATA_DIR = os.path.join(ROOT_DIR, "data")
PUBLICATION_PATH = os.path.join(DATA_DIR, "publication.md")
PUBLICATION_MINI_PATH = os.path.join(DATA_DIR, "publication_mini.md")
PUBLICATIONS_PATH = os.path.join(DATA_DIR, "project_1_publications.json")

VECTOR_DB_DIR = os.path.join(OUTPUTS_DIR, "vector_db")
