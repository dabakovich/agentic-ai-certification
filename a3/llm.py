from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def get_llm(llm_name: str):
    # local_model_name = "gemma3:4b"
    # local_model_name = "llama3.2:3b"
    # local_model_name = "qwen3:4b"
    # llm = ChatOllama(model=local_model_name, temperature=0)

    if llm_name == "gemma3:4b":
        llm = ChatOllama(model=llm_name, temperature=0)

    if llm_name == "gpt-4.1-nano" or llm_name == "gpt-4o-mini":
        llm = ChatOpenAI(model_name=llm_name, temperature=0)

    return llm
