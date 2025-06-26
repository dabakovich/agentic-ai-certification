from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama

from common.utils import load_yaml_config
from paths import MODELS_CONFIG_FPATH


def get_llm(llm: str = "gpt_nano"):
    models_config = load_yaml_config(MODELS_CONFIG_FPATH)
    open_ai_models = models_config.get("openai", {})
    local_models = models_config.get("local", {})

    if llm in open_ai_models:
        return ChatOpenAI(model_name=open_ai_models[llm], temperature=0.0)
    elif llm in local_models:
        return ChatOllama(model=local_models[llm], temperature=0.0)
    else:
        raise ValueError(f"LLM {llm} not supported")
