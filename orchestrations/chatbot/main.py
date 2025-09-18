"""
LangGraph chatbot implementation with streaming capabilities and memory.
"""
from typing import Annotated, Dict, Any

from typing_extensions import TypedDict

from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from orchestrations.llm.openai import model as llm


class State(TypedDict):
    """State definition for the chatbot graph."""
    messages: Annotated[list[Any], add_messages]


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


def run_chat_loop() -> None:
    """Run the interactive chat loop with memory."""
    graph = create_chatbot_graph().compile()

    message_history = []

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            message_history.append({"role": "user", "content": user_input})

            initial_state = {"messages": message_history}

            for event in graph.stream(initial_state):
                for value in event.values():
                    assistant_message = value["messages"][-1]
                    print("Assistant:", assistant_message.content)

                    message_history.append(assistant_message)
        except (KeyboardInterrupt, EOFError):
            print("\nChat session terminated.")
            break
        except Exception as e:
            print(f"Error: {e}")
            break


if __name__ == "__main__":
    run_chat_loop()
