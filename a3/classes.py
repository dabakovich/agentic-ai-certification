from typing import List, Optional

from pydantic import BaseModel


class PromptConfig(BaseModel):
    instruction: str | List[str]
    role: Optional[str] = None
    context: Optional[str] = None
    output_constraints: Optional[str | List[str]] = None
    style_or_tone: Optional[str | List[str]] = None
    output_format: Optional[str | List[str]] = None
    examples: Optional[str | List[str]] = None
    goal: Optional[str] = None
    reasoning_strategy: Optional[str] = None
