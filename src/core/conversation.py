import sys
from llm.client import get_llm
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from common.utils import load_yaml_config
from paths import PROMPT_CONFIG_FPATH
from llm.prompt_builder import build_prompt_from_config
from langchain.chains.conversation.base import ConversationChain
from vector_store import VectorStore
from common.constants import prompt_template
from container import Container
from dependency_injector.wiring import Provide


class Conversation:
    def __init__(
        self,
        llm_name="gpt_nano",
        vector_store: VectorStore = Provide[Container.vector_store],
    ):
        self.llm = get_llm(llm_name)
        self.prompts_config = load_yaml_config(PROMPT_CONFIG_FPATH)
        self.memory = ConversationBufferMemory(
            chat_memory=FileChatMessageHistory("chats/chat_history.json"),
            return_messages=True,
        )
        self.conversation = ConversationChain(llm=self.llm, memory=self.memory)
        self.vector_store = vector_store

    def run(self):
        self.memory.chat_memory.add_message(
            SystemMessage(
                content=build_prompt_from_config(
                    self.prompts_config["assistant_system_message"]
                )
            )
        )
        while True:
            user_input = input("You: ")

            if user_input.lower() == "q":
                break

            prompt = self.build_retrieval_prompt(user_input)

            stream_generator = self.llm.stream(
                self.memory.chat_memory.messages + [HumanMessage(content=prompt)]
            )

            response = ""
            sys.stdout.write("AI: ")
            for chunk in stream_generator:
                response += chunk.content
                sys.stdout.write(f"{chunk.content}")
                sys.stdout.flush()

            sys.stdout.write("\n")

            self.memory.chat_memory.add_message(HumanMessage(content=user_input))
            self.memory.chat_memory.add_message(AIMessage(content=response))

    def build_retrieval_prompt(self, user_input: str) -> str:
        relevant_publications = self.vector_store.retrieve_publications(user_input)

        context = "\n\n---\n\n".join(relevant_publications["documents"])

        if context == "":
            return "No relevant documents found"

        prompt = prompt_template.format(context=context, question=user_input)

        return prompt
