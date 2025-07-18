from types import PromptConfig
from typing import List, Union

from utils import load_config

from paths import REASONING_FILE_PATH

reasoning_strategies = load_config(REASONING_FILE_PATH)["reasoning_strategies"]


def format_prompt_section(
    lead_in: str, value: Union[str, List[str]], get_prefix=None
) -> str:
    if isinstance(value, list):
        if get_prefix:
            formatted_value = "\n".join(
                f"{get_prefix(i)}{item}" for i, item in enumerate(value, 1)
            )
        else:
            formatted_value = "\n".join(f"- {item}" for item in value)
    else:
        formatted_value = value
    return f"{lead_in}\n{formatted_value}"


def build_prompt_body(prompt_config: PromptConfig, input_data: str = "") -> str:
    prompt_parts = []

    if role := prompt_config.role:
        prompt_parts.append(f"You are {role}.")

    if instruction := prompt_config.instruction:
        prompt_parts.append(
            format_prompt_section("Your task is as follows:", instruction)
        )

    if context := prompt_config.context:
        prompt_parts.append(f"Here's some background that may help you:\n{context}")

    if output_constraints := prompt_config.output_constraints:
        prompt_parts.append(
            format_prompt_section(
                "Ensure your response follows these rules:", output_constraints
            )
        )

    if style_or_tone := prompt_config.style_or_tone:
        prompt_parts.append(
            format_prompt_section(
                "Follow these style and tone guidelines in your response:",
                style_or_tone,
            )
        )

    if output_format := prompt_config.output_format:
        prompt_parts.append(
            format_prompt_section("Structure your response as follows:", output_format)
        )

    if examples := prompt_config.examples:
        prompt_parts.append(
            format_prompt_section(
                "Here are some examples to guide your response:",
                examples,
                lambda i: f"Example {i}:\n",
            )
        )

    if goal := prompt_config.goal:
        prompt_parts.append(f"Your goal is to achieve the following outcome:\n{goal}")

    if input_data:
        prompt_parts.append(
            "Here is the content you need to work with:\n"
            "<<<BEGIN CONTENT>>>\n"
            "```\n" + input_data.strip() + "\n```\n<<<END CONTENT>>>"
        )

    if reasoning := prompt_config.reasoning_strategy:
        strategy_prompt = reasoning_strategies.get(reasoning, "")
        if strategy_prompt:
            prompt_parts.append(strategy_prompt.strip())

    if input_data:
        prompt_parts.append("Now perform the task as instructed above.")

    return "\n\n".join(prompt_parts)
