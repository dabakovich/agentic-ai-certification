from constants import NODE
from graphs.a3_graph import build_a3_graph
from states.a3_state import initialize_a3_state
from utils import load_config, load_publication_example


def run_a3_agent(text: str):
    a3_config = load_config()["a3"]

    agents_config = a3_config["agents"]

    initial_state = initialize_a3_state(
        input_text=text,
        manager_config=agents_config[NODE.MANAGER.value]["prompt_config"],
        tldr_config=agents_config[NODE.TLDR_GENERATOR.value]["prompt_config"],
        title_config=agents_config[NODE.TITLE_GENERATOR.value]["prompt_config"],
        reviewer_config=agents_config[NODE.REVIEWER.value]["prompt_config"],
        max_revisions=10,
    )

    graph = build_a3_graph(a3_config)

    final_state = graph.invoke(initial_state)

    return final_state


def main():
    sample_text = load_publication_example(1)
    response = run_a3_agent(sample_text)

    print("=" * 80)
    print("🔍 A3-SYSTEM DEMO")
    print("=" * 80)
    print("Manager brief:")
    print(response["manager_brief"])
    print("=" * 80)
    print("Title:")
    print(response["title"])
    print("=" * 80)
    print("TL;DR:")
    print(response["tldr"])
    print("=" * 80)


if __name__ == "__main__":
    main()
