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

MANAGER_MESSAGES = "manager_messages"
MANAGER_BRIEF = "manager_brief"

TITLE_GEN_MESSAGES = "title_gen_messages"
TLDR_GEN_MESSAGES = "tldr_gen_messages"
REVIEWER_MESSAGES = "reviewer_messages"

TLDR = "tldr"
TITLE = "title"

REVISION_ROUND = "revision_round"
NEEDS_REVISION = "needs_revision"

TLDR_FEEDBACK = "tldr_feedback"
TITLE_FEEDBACK = "title_feedback"

TLDR_APPROVED = "tldr_approved"
TITLE_APPROVED = "title_approved"
