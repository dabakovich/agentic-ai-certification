from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from constants import OPEN_AI_MODELS, LOCAL_MODELS


def get_llm(llm: str = "gpt_nano"):
    if llm in OPEN_AI_MODELS:
        return ChatOpenAI(model_name=OPEN_AI_MODELS[llm], temperature=0.0)
    elif llm in LOCAL_MODELS:
        return ChatOllama(model=LOCAL_MODELS[llm], temperature=0.0)
    else:
        raise ValueError(f"LLM {llm} not supported")
