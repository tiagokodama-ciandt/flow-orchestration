# Flow Orchestration

This project demonstrates the use of LangGraph for creating AI orchestration flows, including a web search agent and a conversational chatbot.

## Project Structure

- `orchestrations/websearch/`: Web search agent using Tavily API
- `orchestrations/chatbot/`: Conversational chatbot with memory
- `orchestrations/llm/`: LLM integration with OpenAI

## Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/flow-orchestration.git
cd flow-orchestration
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory with your API keys:
```
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## How to Run Web Search

The web search module uses LangGraph to create an agent that can search the web using the Tavily API.

### Method 1: Run from Command Line

You can run the web search directly from the command line:

```bash
python -m orchestrations.websearch.main
```

This will execute a search with a default query. To use a custom query, you can modify the sample query in the `main.py` file.

### Method 2: Import in Your Code

You can also use the web search functionality in your own Python code:

```python
from orchestrations.websearch.main import websearch, run_websearch

# Option 1: Get the search result as a string
result = websearch("What are the latest developments in AI?")
print(result)

# Option 2: Use the convenience function that prints the result
run_websearch("What is the current status of Mars exploration?")
```

## How to Run Chatbot

The chatbot module provides a conversational interface with memory using LangGraph.

### Method 1: Run from Command Line

Start an interactive chat session from the command line:

```bash
python -m orchestrations.chatbot.main
```

This will start an interactive chat session where you can type messages and receive responses. Type "exit", "quit", or "q" to end the session.

### Method 2: Import in Your Code

You can also integrate the chatbot into your own Python code:

```python
from orchestrations.chatbot.main import create_chatbot_graph

# Create and compile the chatbot graph
graph = create_chatbot_graph().compile()

# Initialize with an empty message history
message_history = []

# Add a user message
message_history.append({"role": "user", "content": "Hello, how are you?"})

# Process the message and get a response
result = graph.invoke({"messages": message_history})

# Extract and print the assistant's response
assistant_message = result["messages"][-1]
print("Assistant:", assistant_message.content)

# Update the message history for continued conversation
message_history.append(assistant_message)
```

## LLM Integration

Both the web search and chatbot modules use the LLM integration from `orchestrations.llm.openai`. The LLM is configured to use OpenAI's API with specific parameters:

```python
from orchestrations.llm.openai import model as llm

# Use the LLM in your code
response = llm.invoke("Your prompt here")
```

## Advanced Usage

### Customizing the Web Search Agent

You can customize the web search agent by modifying the system prompt in `orchestrations/websearch/main.py`:

```python
SYSTEM_PROMPT = """Your custom instructions here..."""
```

### Extending the Chatbot

To add more capabilities to the chatbot, you can modify the `chatbot` function in `orchestrations/chatbot/main.py` or add additional nodes to the graph.

## License

[Your License Information]