# Web Search Agent with Tavily

This module provides a LangGraph-based agent that can search the web using the Tavily API.

## Setup

1. Make sure you have the required dependencies installed:
   ```
   pip install -r requirements.txt
   ```

2. Set up your environment variables in a `.env` file:
   ```
   OPENAI_API_KEY=your_openai_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

## Usage

You can use the websearch agent in your code:

```python
from orchestrations.websearch.main import websearch

# Perform a web search with a specific query
result = websearch("What are the latest developments in AI?")
print(result)
```

Or you can use the convenience function:

```python
from orchestrations.websearch.main import run_websearch

# Run with default query
run_websearch()

# Run with custom query
run_websearch("What is the current status of Mars exploration?")
```

## Example

See `examples/websearch_example.py` for a complete example of how to use the websearch agent.