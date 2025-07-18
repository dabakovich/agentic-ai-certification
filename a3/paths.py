import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

A3_DIR = os.path.join(ROOT_DIR, "a3")

CONFIG_DIR = os.path.join(A3_DIR, "config")

CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, "config.yaml")

REASONING_FILE_PATH = os.path.join(CONFIG_DIR, "reasoning.yaml")
