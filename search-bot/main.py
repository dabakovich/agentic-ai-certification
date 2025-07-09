from typing import Annotated

from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from llm import llm


class State(BaseModel):
    messages: Annotated[list, add_messages] = []


def get_tools():
    return [TavilySearch(max_results=5, search_depth="advanced")]


def llm_node(state: State):
    tools = get_tools()

    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke(state.messages)

    return {"messages": [response]}


def tools_node(state: State):
    tools = get_tools()
    tool_registry = {tool.name: tool for tool in tools}

    last_message: AIMessage = state.messages[-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        tool = tool_registry[tool_call["name"]]
        result = tool.invoke(tool_call["args"])

        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages}


def should_continue(state: State):
    """Decide whether use tool node or provide a final answer"""
    last_message = state.messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return END


def create_agent():
    graph = StateGraph(State)

    graph.add_node("llm", llm_node)
    graph.add_node("tools", tools_node)

    graph.set_entry_point("llm")

    graph.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")

    return graph.compile()


def main():
    load_dotenv()

    agent = create_agent()

    iniitial_state = State(
        messages=[
            SystemMessage(
                "You are a helpful assistant with access to web search. Use search tool when you need current information."
            ),
            HumanMessage("What weather will be in Lviv, Ukraine on this Sunday?"),
            # HumanMessage("What's the latest news about AI developments in 2025?"),
        ]
    )

    result = agent.invoke(iniitial_state)

    print(result["messages"][-1])


if __name__ == "__main__":
    main()
