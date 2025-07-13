from pydantic import BaseModel
from langgraph.graph import StateGraph, END


class State(BaseModel):
    document: str | None = None
    gazetteer_result: list[str] = []
    ner_result: list[str] = []
    llm_result: list[str] = []
    final_result: list[str] = []


def gazeteer_extraction_node(state: State):
    print("Making gazeteer tags...")
    return {"gazetteer_result": ["a"]}


def ner_extraction_node(state: State):
    print("Making NER tags...")
    return {"ner_result": ["b"]}


def llm_extraction_node(state: State):
    print("Making LLM tags...")
    return {"llm_result": ["c"]}


def aggregation_node(state: State):
    print("In the union node")
    if state.gazetteer_result and state.ner_result and state.llm_result:
        return {
            "final_result": state.gazetteer_result + state.ner_result + state.llm_result
        }


def final_node(state: State):
    print("In the final node")
    if state.final_result:
        print(state.final_result)
    else:
        print("No tags were extracted")


def router(state: State):
    if state.document:
        return ["gazeteer_extraction", "ner_extraction", "llm_extraction"]

    return "final_node"


def create_agent():
    graph = StateGraph(State)

    graph.add_node("gazeteer_extraction", gazeteer_extraction_node)
    graph.add_node("ner_extraction", ner_extraction_node)
    graph.add_node("llm_extraction", llm_extraction_node)
    graph.add_node("aggregation", aggregation_node)
    graph.add_node("final_node", final_node)

    graph.set_conditional_entry_point(
        router,
        {
            "gazeteer_extraction": "gazeteer_extraction",
            "ner_extraction": "ner_extraction",
            "llm_extraction": "llm_extraction",
            "final_node": "final_node",
        },
    )

    graph.add_edge(
        ["gazeteer_extraction", "ner_extraction", "llm_extraction"], "aggregation"
    )
    graph.add_edge("aggregation", "final_node")
    graph.add_edge("final_node", END)

    return graph.compile()


def main():
    agent = create_agent()

    agent.invoke(State(document="Hey"))


if __name__ == "__main__":
    main()
