from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

# local_model_name = "gemma3:4b"
# local_model_name = "llama3.2:3b"
local_model_name = "qwen3:4b"
llm = ChatOllama(model=local_model_name, temperature=0)

# remote_model_name = "gpt-4.1-nano"
remote_model_name = "gpt-4o-mini"
# llm = ChatOpenAI(model_name=remote_model_name, temperature=0.0)
