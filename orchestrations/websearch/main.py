"""
LangGraph websearch agent implementation with Tavily integration.
"""
from typing import Annotated, List, TypedDict, Union

from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from orchestrations.llm.openai import model as llm
from orchestrations.websearch.tavily import search_tool


class State(TypedDict):
    """State definition for the websearch agent graph."""
    messages: Annotated[List[Union[
        HumanMessage, AIMessage, SystemMessage, ToolMessage
    ]], add_messages]
    next: str


# Define the system prompt
SYSTEM_PROMPT = """You are a helpful assistant that can search the web 
for information.
Use the search tool when you need to look up information on the internet.
Always cite your sources when providing information from web searches.
Provide a concise summary of the search results.
"""


def create_agent_node():
    """Create the agent node that decides whether to use tools or respond."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="messages"),
    ])

    return prompt | llm.bind_tools([search_tool])


def agent_node(state: State) -> State:
    agent = create_agent_node()  # This is your pipeline
    # Run the agent with the current messages
    response = agent.invoke({"messages": state["messages"]})
    # Append the new message to the state, ensuring correct type
    if isinstance(response, (
        HumanMessage, AIMessage, SystemMessage, ToolMessage
    )):
        state["messages"].append(response)
    else:
        raise TypeError(f"Unexpected message type: {type(response)}")
    return state


def should_continue(state: State) -> str:
    """Determine if the agent should continue or finish."""
    messages = state["messages"]
    last_message = messages[-1]

    # If the last message is from the AI and doesn't request a tool, we're done
    if isinstance(last_message, AIMessage) and not last_message.tool_calls:
        return "end"

    # If the last message is a tool message, continue to the agent
    if isinstance(last_message, ToolMessage):
        return "agent"

    # Otherwise, continue to the agent
    return "agent"


def create_websearch_graph() -> StateGraph:
    """
    Create and configure the websearch agent graph.

    Returns:
        StateGraph: The compiled graph ready for execution.
    """
    # Create the graph
    graph_builder = StateGraph(State)

    # Add nodes
    graph_builder.add_node("agent", agent_node)
    graph_builder.add_node("tools", ToolNode([search_tool]))

    # Add edges
    graph_builder.add_edge(START, "agent")
    graph_builder.add_conditional_edges(
        "agent",
        should_continue,
        {
            "agent": "tools",
            "end": END,
        },
    )
    graph_builder.add_edge("tools", "agent")

    # Set the entry point
    graph_builder.set_entry_point("agent")

    return graph_builder


def websearch(query: str) -> str:
    """
    Perform a web search with the given query and return the result.

    Args:
        query: The search query string

    Returns:
        str: The search result summary
    """
    graph = create_websearch_graph().compile()

    # Initialize with system message and query
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query)
    ]

    initial_state = {"messages": messages, "next": ""}

    # Execute the graph
    final_state = graph.invoke(initial_state)

    # Extract the final AI response
    final_messages = final_state["messages"]
    for message in reversed(final_messages):
        if isinstance(message, AIMessage):
            return message.content

    return "No response generated."


def run_websearch(query: str) -> None:
    """
    Run a web search with the given query and print the result.

    Args:
        query: The search query string (defaults to a sample query)
    """
    print(f"Searching for: {query}")
    result = websearch(query)
    print("\nSearch Result:")
    print(result)


if __name__ == "__main__":
    # Example usage with a hard-coded query
    sample_query = "Crie uma história usando o meme mais atual do Brasil?"
    run_websearch(sample_query)
