from typing import List

from classes import Joke


def joke_reducer(current: List[Joke], new: List[Joke]) -> List[Joke]:
    # Clear jokes if we're providing empty list
    if not new:
        return []

    # Just concatenate jokes if we're providing new joke
    return current + new
