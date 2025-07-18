from typing import Any, Callable, Dict, Literal

from constants import FIELD, NODE
from langchain_core.messages import HumanMessage
from states.a3_state import A3State

from llm import get_llm

from .output_types import ReviewOutput

NodeType = Callable[[A3State], Dict[str, Any]]


def make_manager_node(llm_model: str) -> NodeType:
    llm = get_llm(llm_model)

    def manager_node(state: A3State) -> Dict[str, Any]:
        ai_response = llm.invoke(state.manager_messages)
        content = f"This is your manager's brief for your review:\n\n{ai_response.content.strip()}\n\n"
        human_message = HumanMessage(content)

        return {
            FIELD.MANAGER_MESSAGES.value: [ai_response],
            FIELD.MANAGER_BRIEF.value: [ai_response],
            FIELD.TITLE_GEN_MESSAGES.value: [human_message],
            FIELD.TLDR_GEN_MESSAGES.value: [human_message],
            FIELD.REVIEWER_MESSAGES.value: [human_message],
        }

    return manager_node


def make_tldr_generator_node(llm_model: str) -> NodeType:
    llm = get_llm(llm_model)

    def tldr_generator_node(state: A3State) -> Dict[str, Any]:
        if state.tldr_approved is True:
            print("🎯 TL;DR Generator: Already approved, skipping...")
            return {}

        print("🎯 TL;DR Generator: Creating TL;DR...")
        messages = state.title_gen_messages
        reviewer_message = HumanMessage(
            f"Following is the review from your reviewer:\n\n {state.tldr_feedback or 'No feedback provided'}\n\n"
        )

        messages += [
            reviewer_message,
            HumanMessage(
                "Proceed with your TL;DR generation using latest feedback (if any)."
            ),
        ]

        ai_response = llm.invoke(messages)
        content = ai_response.content.strip()

        return {
            FIELD.TLDR_GEN_MESSAGES.value: [messages[-1], ai_response],
            FIELD.TLDR.value: content,
            FIELD.TLDR_FEEDBACK.value: "",
        }

    return tldr_generator_node


def make_title_generator_node(llm_model: str) -> NodeType:
    llm = get_llm(llm_model)

    def title_generator_node(state: A3State) -> Dict[str, Any]:
        if state.title_approved is True:
            print("🎯 Title Generator: Already approved, skipping...")
            return {}

        print("🎯 Title Generator: Creating title...")
        messages = state.title_gen_messages
        reviewer_message = HumanMessage(
            f"Following is the review from your reviewer:\n\n {state.title_feedback or 'No feedback provided'}\n\n"
        )

        messages += [
            reviewer_message,
            HumanMessage(
                "Proceed with your title generation using latest feedback (if any)."
            ),
        ]

        ai_response = llm.invoke(messages)
        content = ai_response.content.strip()

        return {
            FIELD.TITLE_GEN_MESSAGES.value: [messages[-1], ai_response],
            FIELD.TITLE.value: content,
            FIELD.TITLE_FEEDBACK.value: "",
        }

    return title_generator_node


def make_reviewer_node(llm_model: str) -> NodeType:
    llm = get_llm(llm_model)

    def reviewer_node(state: A3State) -> Dict[str, Any]:
        """
        Reviewer node that processes the input text and generates feedback.
        """

        if state.revision_round >= state.max_revisions:
            print(
                "🔒 Reviewer: Maximum revisions reached, forcing approval for all components."
            )
            return {
                FIELD.NEEDS_REVISION.value: False,
                FIELD.TITLE_APPROVED.value: True,
                FIELD.TLDR_APPROVED.value: True,
            }

        print("📝 Reviewer: Generating feedback...")

        review_input = f"""
# TL;DR:\n
{state.tldr or "Not generated"}\n{"-" * 20}\n
# Title:\n
{state.title or "Not generated"}\n{"-" * 20}\n
        """

        messages = state.reviewer_messages + [
            HumanMessage(
                f"Please review the following content and provide feedback:\n\n{review_input}\n\n"
                "If you have any specific feedback for the TL;DR, title, or references, please include it."
            )
        ]
        response = llm.with_structured_output(ReviewOutput).invoke(messages)
        revision_round = state.revision_round + 1

        overall_approved = response.tldr_approved and response.title_approved

        print(f"✅ Review completed: approved = {overall_approved}")
        print(f"📋 Feedback: {response.model_dump()}")

        # Show individual component status
        components_status = [
            f"TL;DR: {'✅' if response.tldr_approved else '❌'}",
            f"Title: {'✅' if response.title_approved else '❌'}",
        ]
        print(f"📊 Component Status: {' | '.join(components_status)}")

        if not overall_approved:
            return {
                FIELD.NEEDS_REVISION.value: True,
                FIELD.REVISION_ROUND.value: revision_round,
                FIELD.TLDR_FEEDBACK.value: response.tldr_feedback,
                FIELD.TITLE_FEEDBACK.value: response.title_feedback,
                FIELD.TLDR_APPROVED.value: response.tldr_approved,
                FIELD.TITLE_APPROVED.value: response.title_approved,
            }
        else:
            print("✅ All components approved - proceeding to final output")

            return {
                FIELD.NEEDS_REVISION.value: False,
                FIELD.REVISION_ROUND.value: revision_round,
                FIELD.TLDR_FEEDBACK.value: response.tldr_feedback,
                FIELD.TITLE_FEEDBACK.value: response.title_feedback,
                FIELD.TLDR_APPROVED.value: response.tldr_approved,
                FIELD.TITLE_APPROVED.value: response.title_approved,
            }

    return reviewer_node


def route_from_reviewer(state: A3State) -> Literal["revision_dispatcher", "end"]:
    """
    Conditional routing function that determines whether to dispatch revisions or end.
    """

    if not state.needs_revision:
        print("✅ All components approved - routing to END")
        return "end"
    else:
        print("🔄 Some components need revision - routing to revision dispatcher")
        return [
            NODE.TLDR_GENERATOR.value,
            NODE.TITLE_GENERATOR.value,
        ]
