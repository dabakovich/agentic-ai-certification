import sys
from llm import get_llm
from langchain.schema import SystemMessage, HumanMessage, AIMessage


class Conversation:
    def __init__(self, llm_name="gpt_nano"):
        self.llm = get_llm(llm_name)

    def run(self):
        messages = [
            SystemMessage(content="You are a helpful AI assistant. Answer very shortly")
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
