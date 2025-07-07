from langchain.prompts import PromptTemplate
from langchain.schema import SystemMessage

system_message = SystemMessage(
    """You are a basic joke generator.
You will be providen category and language.
Answer only with joke text, without any formatting and additional text.
Don't provide translations if you generated joke in different language than English.
You will be providen your previous jokes, don't repeat them."""
)

joke_prompt = PromptTemplate(
    input_variables=["category", "language"],
    template="""
Category: {category}

Language: {language}

Previous jokes:
{prev_jokes}
""",
)
