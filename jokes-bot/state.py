from typing import Annotated, List
from pydantic import BaseModel
from classes import Joke
from reducers import joke_reducer
from constants import categories


class JokeState(BaseModel):
    jokes: Annotated[List[Joke], joke_reducer] = []
    user_choice: str = "next_joke"
    category: str = categories[0]
    language: str = "en"
    quit: bool = False
    joke_for_review: Joke | None = None
    rejected_jokes: Annotated[List[Joke], joke_reducer] = []
    approved: bool = False
