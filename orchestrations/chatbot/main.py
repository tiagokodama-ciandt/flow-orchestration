"""
LangGraph chatbot implementation with streaming capabilities.
"""
from typing import Annotated, Dict, Any

from typing_extensions import TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from orchestrations.llm.openai import model as llm


class State(TypedDict):
    """State definition for the chatbot graph."""
    messages: Annotated[list[Any], add_messages]


# Define the chatbot node function
def chatbot(state: State) -> Dict[str, Any]:
    """Process user messages and generate a response."""
    return {"messages": [llm.invoke(state["messages"])]}


def create_chatbot_graph() -> StateGraph:
    """
    Create and configure the chatbot graph.

    Returns:
        StateGraph: The compiled graph ready for execution.
    """
    graph_builder = StateGraph(State)

    # Configure the graph
    graph_builder.add_node("chatbot", chatbot, input_schema=State)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)

    return graph_builder


def stream_graph_updates(graph: Any, user_input: str) -> None:
    """
    Stream updates from the graph execution.

    Args:
        graph: The compiled graph to execute
        user_input: The user's message
    """
    initial_state = {"messages": [{"role": "user", "content": user_input}]}

    for event in graph.stream(initial_state):
        for value in event.values():
            print("Assistant:", value["messages"][-1].content)


def run_chat_loop() -> None:
    """Run the interactive chat loop."""
    graph = create_chatbot_graph().compile()

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(graph, user_input)
        except (KeyboardInterrupt, EOFError):
            print("\nChat session terminated.")
            break
        except Exception as e:
            print(f"Error: {e}")
            # Fallback if input() is not available
            fallback_input = "What do you know about LangGraph?"
            print(f"User: {fallback_input}")
            stream_graph_updates(graph, fallback_input)
            break


if __name__ == "__main__":
    run_chat_loop()
