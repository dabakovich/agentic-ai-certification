from typing import Optional, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph.message import AnyMessage, add_messages

from pydantic import BaseModel
from types import PromptConfig
from prompt_builder import build_prompt_body


class A3State(BaseModel):
    """State class for A3 agentic system"""

    input_text: str

    manager_messages: Optional[str]

    title_gen_messages: Annotated[list[AnyMessage], add_messages]
    tldr_gen_messages: Annotated[list[AnyMessage], add_messages]

    tldr: Optional[str]
    title: Optional[str]

    # Revision info
    revision_round: Optional[int]
    needs_revision: Optional[bool]
    tldr_feedback: Optional[str]
    title_feedback: Optional[str]

    # Personal feedback
    tldr_approved: Optional[bool]
    title_approved: Optional[bool]

    max_revisions: Optional[int]


def initialize_a3_state(
    input_text: str,
    manager_config: PromptConfig,
    title_config: PromptConfig,
    tldr_config: PromptConfig,
    reviewer_config: PromptConfig,
    max_revisions: int,
) -> A3State:
    manager_messages = [
        SystemMessage(build_prompt_body(manager_config)),
        # Think to move it into node
        SystemMessage(f"Here's your input text:\n\n{input_text}"),
    ]
    title_gen_messages = [
        SystemMessage(build_prompt_body(title_config)),
        # Think to move it into node
        SystemMessage(f"Here's your input text for title generation:\n\n{input_text}"),
    ]
    tldr_gen_messages = [
        SystemMessage(build_prompt_body(tldr_config)),
        # Think to move it into node
        SystemMessage(f"Here's your input text for TL;DR generation:\n\n{input_text}"),
    ]

    reviewer_messages = [
        SystemMessage(build_prompt_body(reviewer_config)),
        # Think to move it into node
        SystemMessage(f"Here's your input text for review work:\n\n{input_text}"),
    ]

    return A3State(
        input_text=input_text,
        manager_messages=manager_messages,
        title_gen_messages=title_gen_messages,
        tldr_gen_messages=tldr_gen_messages,
        reviewer_messages=reviewer_messages,
        title=None,
        tldr=None,
        revision_round=0,
        needs_revision=False,
        tldr_feedback=None,
        title_feedback=None,
        tldr_approved=False,
        title_approved=False,
        max_revisions=max_revisions,
    )
