from enum import Enum


class ROLE(Enum):
    SYSTEM = "system"
    AI = "ai"
    HUMAN = "human"


# NODES

MANAGER = "manager"
TITLE_GENERATOR = "title_generator"
TLDR_GENERATOR = "tldr_generator"
REVIEWER = "reviewer"


# FIELDS

INPUT_TEXT = "input_text"


REVISION_ROUND = "revision_round"
