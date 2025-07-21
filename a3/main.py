from constants import MANAGER, REVIEWER, TITLE_GENERATOR, TLDR_GENERATOR
from graphs.a3_graph import build_a3_graph
from states.a3_state import initialize_a3_state
from utils import load_config, load_publication_example
from dotenv import load_dotenv
from states.a3_state import A3State


def run_a3_agent(text: str):
    a3_config = load_config()["a3"]

    agents_config = a3_config["agents"]

    initial_state = initialize_a3_state(
        input_text=text,
        manager_config=agents_config[MANAGER]["prompt_config"],
        tldr_config=agents_config[TLDR_GENERATOR]["prompt_config"],
        title_config=agents_config[TITLE_GENERATOR]["prompt_config"],
        reviewer_config=agents_config[REVIEWER]["prompt_config"],
        max_revisions=10,
    )

    graph = build_a3_graph(a3_config)

    final_state = graph.invoke(initial_state)

    return final_state


def main():
    load_dotenv(override=True)

    sample_text = load_publication_example(1)
    response = run_a3_agent(sample_text)

    state = A3State.model_validate(response)

    print("=" * 80)
    print("🔍 A3-SYSTEM DEMO")
    print("=" * 80)
    print("Manager brief:")
    print(state.manager.brief)
    print("=" * 80)
    print("Title:")
    print(state.title_generator.draft)
    print("=" * 80)
    print("TL;DR:")
    print(state.tldr_generator.draft)
    print("=" * 80)


if __name__ == "__main__":
    main()
