from typing import Union, List, Dict, Any


def build_prompt_from_config(
    config: Dict[str, Any],
    input_data: str = "",
) -> str:
    """Builds a complete prompt string based on a config dictionary.

    Args:
        config: Dictionary specifying prompt components.
        input_data: Content to be summarized or processed.

    Returns:
        A fully constructed prompt as a string.

    Raises:
        ValueError: If the required 'instruction' field is missing.
    """
    prompt_parts = []

    if role := config.get("role"):
        prompt_parts.append(f"You are {lowercase_first_char(role.strip())}.")

    instruction = config.get("instruction")
    if not instruction:
        raise ValueError("Missing required field: 'instruction'")
    prompt_parts.append(format_prompt_section("Your task is as follows:", instruction))

    if context := config.get("context"):
        prompt_parts.append(f"Here’s some background that may help you:\n{context}")

    if constraints := config.get("output_constraints"):
        prompt_parts.append(
            format_prompt_section(
                "Ensure your response follows these rules:", constraints
            )
        )

    if tone := config.get("style_or_tone"):
        prompt_parts.append(
            format_prompt_section(
                "Follow these style and tone guidelines in your response:", tone
            )
        )

    if format_ := config.get("output_format"):
        prompt_parts.append(
            format_prompt_section("Structure your response as follows:", format_)
        )

    if examples := config.get("examples"):
        prompt_parts.append("Here are some examples to guide your response:")
        if isinstance(examples, list):
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"Example {i}:\n{example}")
        else:
            prompt_parts.append(str(examples))

    if goal := config.get("goal"):
        prompt_parts.append(f"Your goal is to achieve the following outcome:\n{goal}")

    if input_data:
        prompt_parts.append(
            "Here is the content you need to work with:\n"
            "<<<BEGIN CONTENT>>>\n"
            "```\n" + input_data.strip() + "\n```\n<<<END CONTENT>>>"
        )

    prompt_parts.append("Now perform the task as instructed above.")
    return "\n\n".join(prompt_parts)


def lowercase_first_char(text: str) -> str:
    """Lowercases the first character of a string.

    Args:
        text: Input string.

    Returns:
        The input string with the first character lowercased.
    """
    return text[0].lower() + text[1:] if text else text


def format_prompt_section(lead_in: str, value: Union[str, List[str]]) -> str:
    """Formats a prompt section by joining a lead-in with content.

    Args:
        lead_in: Introduction sentence for the section.
        value: Section content, as a string or list of strings.

    Returns:
        A formatted string with the lead-in followed by the content.
    """
    if isinstance(value, list):
        formatted_value = "\n".join(f"- {item}" for item in value)
    else:
        formatted_value = value
    return f"{lead_in}\n{formatted_value}"
