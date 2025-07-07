from pydantic import BaseModel


class Joke(BaseModel):
    text: str
    category: str
