from pydantic import BaseModel
from langgraph.graph import StateGraph, END


class State(BaseModel):
    document: str | None = None
    gazetteer_result: list[str] = []
    ner_result: list[str] = []
    llm_result: list[str] = []
    final_result: list[str] = []


def planner_node(state: State):
    print("Planing next steps...")


def gazeteer_result_node(state: State):
    print("Making gazeteer tags...")
    return {"gazetteer_result": ["a"]}


def ner_result_node(state: State):
    print("Making NER tags...")
    return {"ner_result": ["b"]}


def llm_result_node(state: State):
    print("Making LLM tags...")
    return {"llm_result": ["c"]}


def union_node(state: State):
    print("In the union node")
    if state.gazetteer_result and state.ner_result and state.llm_result:
        return {
            "final_result": state.gazetteer_result + state.ner_result + state.llm_result
        }


def final_node(state: State):
    print("In the final node")
    if state.final_result:
        print(state.final_result)

    print("No tags were extracted")


def router(state: State):
    if state.document:
        return ["gazeteer_result_node", "ner_result_node", "llm_result_node"]

    return "final_node"


def create_agent():
    graph = StateGraph(State)

    graph.add_node("planner_node", planner_node)
    graph.add_node("gazeteer_result_node", gazeteer_result_node)
    graph.add_node("ner_result_node", ner_result_node)
    graph.add_node("llm_result_node", llm_result_node)
    graph.add_node("union_node", union_node)
    graph.add_node("final_node", final_node)

    graph.set_entry_point("planner_node")

    graph.add_conditional_edges(
        "planner_node",
        router,
        {
            "gazeteer_result_node": "gazeteer_result_node",
            "ner_result_node": "ner_result_node",
            "llm_result_node": "llm_result_node",
            "final_node": "final_node",
        },
    )

    graph.add_edge(
        ["gazeteer_result_node", "ner_result_node", "llm_result_node"], "union_node"
    )
    graph.add_edge("union_node", "final_node")
    graph.add_edge("final_node", END)

    return graph.compile()


def main():
    agent = create_agent()

    agent.invoke(State())


if __name__ == "__main__":
    main()
