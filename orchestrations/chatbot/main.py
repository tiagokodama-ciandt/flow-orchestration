from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_core.messages import HumanMessage

from orchestrations.llm.openai import model as llm

workflow = StateGraph(state_schema=MessagesState)


def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}

workflow.add_node("model", call_model)
workflow.add_edge(START, "model")
workflow.add_edge("model", END)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm Bob."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
output["messages"][-1].pretty_print()


# Run python -m orchestrations.chatbot.main