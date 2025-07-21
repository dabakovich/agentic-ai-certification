from typing import Any, Callable, Dict, Literal

from constants import (
    REVISION_ROUND,
    TITLE_GENERATOR,
    TLDR_GENERATOR,
)
from langchain_core.messages import HumanMessage, AIMessage
from states.a3_state import A3State, Reviewable

from llm import get_llm

from .output_types import ReviewOutput

NodeType = Callable[[A3State], Dict[str, Any]]


def make_manager_node(llm_model: str) -> NodeType:
    llm = get_llm(llm_model)

    def manager_node(state: A3State) -> Dict[str, Any]:
        ai_response = llm.invoke(state.manager.messages)
        content = f"This is your manager's brief for your review:\n\n{ai_response.content.strip()}\n\n"
        human_message = HumanMessage(content)

        return {
            "manager": {
                "messages": state.manager.messages + [ai_response],
                "brief": content,
            },
            TITLE_GENERATOR: {
                "messages": state.title_generator.messages + [human_message]
            },
            TLDR_GENERATOR: {
                "messages": state.tldr_generator.messages + [human_message]
            },
            "reviewer": {"messages": state.reviewer.messages + [human_message]},
        }

    return manager_node


def make_reviewable_generator_node(
    llm_model: str, reviewable_name: Literal["title_generator", "tldr_generator"]
) -> NodeType:
    llm = get_llm(llm_model)

    def reviewable_generator_node(state: A3State) -> Dict[str, Any]:
        reviewable: Reviewable = getattr(state, reviewable_name)

        if reviewable.status == "approved":
            print(f"🎯 {reviewable_name}: Already approved, skipping...")
            return {}

        print(f"🎯 {reviewable_name}: Generating...")
        messages = reviewable.messages
        reviewer_message = HumanMessage(
            f"Following is the review from your reviewer:\n\n {reviewable.feedback or 'No feedback provided'}\n\n"
        )

        messages += [
            reviewer_message,
            HumanMessage(
                "Proceed with your generation using latest feedback (if any)."
            ),
        ]

        ai_response = llm.invoke(messages)
        content = ai_response.content.strip()

        return {
            reviewable_name: {
                "messages": messages + [ai_response],
                "draft": content,
                "status": "needs_revision",
                "feedback": "",
            },
        }

    return reviewable_generator_node


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
                "reviewable": {
                    TITLE_GENERATOR: {
                        "status": "approved",
                    },
                    TLDR_GENERATOR: {"status": "approved"},
                },
            }

        print("📝 Reviewer: Generating feedback...")

        review_input = f"""
# TL;DR:\n
{state.tldr_generator.draft or "Not generated"}\n{"-" * 20}\n
# Title:\n
{state.title_generator.draft or "Not generated"}\n{"-" * 20}\n
        """

        messages = state.reviewer.messages + [
            HumanMessage(
                f"Please review the following content and provide feedback:\n\n{review_input}\n\n"
                "If you have any specific feedback for the TL;DR, title, or references, please include it."
            )
        ]
        response: ReviewOutput = llm.with_structured_output(ReviewOutput).invoke(
            messages
        )
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

        ai_message = AIMessage(response.model_dump_json())

        reviewables_state = {
            TITLE_GENERATOR: {
                "draft": state.title_generator.draft,
                "feedback": response.title_feedback,
                "status": "approved" if response.title_approved else "needs_revision",
            },
            TLDR_GENERATOR: {
                "draft": state.tldr_generator.draft,
                "feedback": response.tldr_feedback,
                "status": "approved" if response.tldr_approved else "needs_revision",
            },
        }

        if not overall_approved:
            return {
                **reviewables_state,
                "reviewer": {"messages": [ai_message]},
                REVISION_ROUND: revision_round,
            }
        else:
            print("✅ All components approved - proceeding to final output")

            return {
                **reviewables_state,
                "reviewer": {"messages": [ai_message]},
                REVISION_ROUND: revision_round,
            }

    return reviewer_node


def route_from_reviewer(state: A3State) -> Literal["revision_dispatcher", "end"]:
    """
    Conditional routing function that determines whether to dispatch revisions or end.
    """

    # If all components are approved, we can end the process
    if all(
        reviewable.status == "approved"
        for reviewable in [state.title_generator, state.tldr_generator]
    ):
        print("✅ All components approved - routing to END")
        return "end"
    else:
        print("🔄 Some components need revision - routing to revision dispatcher")

        # Components that need regeneration
        components_to_regenerate = [
            component
            for component, reviewable in [
                (TITLE_GENERATOR, state.title_generator),
                (TLDR_GENERATOR, state.tldr_generator),
            ]
            if reviewable.status == "needs_revision"
        ]

        return components_to_regenerate
