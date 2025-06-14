from utils import load_env, load_publication, load_yaml_config, get_llm
from prompt_builder import build_system_prompt_from_config, build_rag_assistant_prompt
from paths import PROMPT_CONFIG_FPATH
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from rag import retrieve_relevant_documents
import chromadb

def run_conversation(collection: chromadb.Collection) -> None:
    llm = get_llm(llm="gpt")
    # llm = get_llm(llm="ollama")
    
    # Load prompt configurations
    prompt_configs = load_yaml_config(PROMPT_CONFIG_FPATH)
    system_prompt_config = prompt_configs.get("rag_assistant_prompt")

    system_prompt = build_system_prompt_from_config(system_prompt_config)

    # Save to a file specific to this user/session
    memory = ConversationBufferMemory(
        chat_memory=FileChatMessageHistory("chats/chat_history.json"),
        return_messages=True
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() == "q":
            break

        relevant_documents = retrieve_relevant_documents(collection, user_input)

        messages = [SystemMessage(content=system_prompt)]
        messages.extend(memory.chat_memory.messages)
        messages.append(HumanMessage(content=user_input))

        # If relevant_documents are not empty, add them at the end of the messages
        if relevant_documents:
            messages.append(AIMessage(content=build_rag_assistant_prompt(relevant_documents)))

        response = llm.invoke(messages)
        print("AI: " + response.content)

        # Save user input, relevant documents and response to memory
        memory.chat_memory.add_messages([
            HumanMessage(content=user_input),
            AIMessage(content=build_rag_assistant_prompt(relevant_documents)),
            AIMessage(content=response.content)
        ])