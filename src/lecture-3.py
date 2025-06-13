from utils import load_env, load_publication, load_yaml_config, get_llm
from prompt_builder import build_system_prompt_from_config
from paths import PROMPT_CONFIG_FPATH
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import FileChatMessageHistory
from langchain.schema import SystemMessage, HumanMessage, AIMessage

def run_conversation(publication_content: str, system_prompt_config_name: str) -> None:
    llm = get_llm(llm="ollama")
    
    # Load prompt configurations
    prompt_configs = load_yaml_config(PROMPT_CONFIG_FPATH)
    system_prompt_config = prompt_configs.get(system_prompt_config_name)

    system_prompt = build_system_prompt_from_config(
        system_prompt_config, 
        publication_content
    )

    # Save to a file specific to this user/session
    memory = ConversationBufferMemory(
        chat_memory=FileChatMessageHistory("chats/chat_history.json"),
        return_messages=True
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() == "q":
            break

        messages = [SystemMessage(content=system_prompt)]
        messages.extend(memory.chat_memory.messages)
        messages.append(HumanMessage(content=user_input))

        response = llm.invoke(messages)
        print("AI: " + response.content)

        memory.save_context({"input": user_input}, {"output": response.content})

def main() -> None:
    load_env()
    publication_content = load_publication()
    
    run_conversation(publication_content, "ai_assistant_system_prompt_basic")


if __name__ == "__main__":
    main()