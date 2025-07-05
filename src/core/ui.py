import inquirer

from common.utils import get_llm_choices
from core.conversation import Conversation
from dependency_injector.wiring import inject


@inject
def launch_conversation():
    llm_choices = get_llm_choices()

    # Ask which LLM to use
    answer = inquirer.prompt(
        [
            inquirer.List(
                "llm_name",
                message="Which LLM do you want to use?",
                choices=[choice["name"] for choice in llm_choices],
            )
        ]
    )

    conversation = Conversation(llm_name=answer["llm_name"])
    conversation.run()
