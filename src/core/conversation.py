import sys
from llm.client import get_llm
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from common.utils import load_yaml_config
from paths import PROMPT_CONFIG_FPATH
from llm.prompt_builder import build_prompt_from_config


class Conversation:
    def __init__(self, llm_name="gpt_nano"):
        self.llm = get_llm(llm_name)
        self.prompts_config = load_yaml_config(PROMPT_CONFIG_FPATH)

    def run(self):
        messages = [
            SystemMessage(
                content=build_prompt_from_config(
                    self.prompts_config["assistant_system_message"]
                )
            )
        ]
        while True:
            user_input = input("You: ")

            if user_input.lower() == "q":
                break

            messages.append(HumanMessage(content=user_input))
            stream_generator = self.llm.stream(messages)

            response = ""
            sys.stdout.write("AI: ")
            for chunk in stream_generator:
                response += chunk.content
                sys.stdout.write(f"{chunk.content}")
                sys.stdout.flush()

            sys.stdout.write("\n")

            messages.append(AIMessage(content=response))
