from typing import Annotated, Generic, Literal, Optional, TypeVar, Dict

from classes import PromptConfig
from langchain_core.messages import SystemMessage
from langgraph.graph.message import AnyMessage, add_messages
from prompt_builder import build_prompt_body
from pydantic import BaseModel


class Conversation(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = []


class Manager(Conversation, BaseModel):
    brief: Optional[str] = None


T = TypeVar("T")


class Reviewable(Conversation, BaseModel, Generic[T]):
    # Will be the final version when approved
    draft: Optional[T] = None
    status: Literal["pending", "approved", "needs_revision"] = "pending"
    feedback: Optional[str] = None


class A3State(BaseModel):
    """State class for A3 agentic system"""

    input_text: str

    manager: Manager = Manager()

    title_generator: Reviewable = Reviewable[str]()
    tldr_generator: Reviewable = Reviewable[str]()

    reviewer: Conversation = Conversation()

    # Revision info
    revision_round: Optional[int] = 0

    max_revisions: Optional[int]


def initialize_a3_state(
    input_text: str,
    manager_config: PromptConfig,
    title_config: PromptConfig,
    tldr_config: PromptConfig,
    reviewer_config: PromptConfig,
    max_revisions: int,
) -> A3State:
    a3_state = A3State(
        input_text=input_text,
        max_revisions=max_revisions,
    )

    a3_state.manager.messages = [
        SystemMessage(build_prompt_body(PromptConfig.model_validate(manager_config))),
        # Think to move it into node
        SystemMessage(f"Here's your input text:\n\n{input_text}"),
    ]
    a3_state.title_generator.messages = [
        SystemMessage(build_prompt_body(PromptConfig.model_validate(title_config))),
        # Think to move it into node
        SystemMessage(f"Here's your input text for title generation:\n\n{input_text}"),
    ]
    a3_state.tldr_generator.messages = [
        SystemMessage(build_prompt_body(PromptConfig.model_validate(tldr_config))),
        # Think to move it into node
        SystemMessage(f"Here's your input text for TL;DR generation:\n\n{input_text}"),
    ]

    a3_state.reviewer.messages = [
        SystemMessage(build_prompt_body(PromptConfig.model_validate(reviewer_config))),
        # Think to move it into node
        SystemMessage(f"Here's your input text for review work:\n\n{input_text}"),
    ]

    return a3_state
