from pydantic import BaseModel, Field


class ReviewOutput(BaseModel):
    tldr_approved: bool = Field(description="Whether the TLDR summary is approved")
    tldr_feedback: str = Field(description="Specific feedback for the TLDR summary")
    title_approved: bool = Field(description="Whether the title is approved")
    title_feedback: str = Field(description="Specific feedback for the title")
