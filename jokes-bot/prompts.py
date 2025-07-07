from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage

writer_system_message = SystemMessage(
    """You are a basic joke generator.
You will be providen category and language.
Answer only with joke text, without any formatting and additional text.
Don't provide translations if you generated joke in different language than English.
You will be providen your previous jokes, don't repeat them."""
)

writer_prompt = PromptTemplate(
    input_variables=["category", "language", "prev_jokes", "rejected_jokes"],
    template="""
Category: {category}

Language: {language}

Previous jokes:
{prev_jokes}

Rejected jokes:
{rejected_jokes}
""",
)

critic_system_message = SystemMessage(
    """You are joke critic.
Send a float number between 0 and 1 how funny is the joke.

Answer only with the number, without any additional text.
"""
)

critic_prompt = PromptTemplate(
    input_variables=["joke"],
    template="""
Joke: "{joke}"
""",
)
