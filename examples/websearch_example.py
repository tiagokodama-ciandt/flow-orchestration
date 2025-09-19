"""
Example script demonstrating the websearch agent with a hard-coded query.
"""
from orchestrations.websearch.main import run_websearch

if __name__ == "__main__":
    # Example with default query
    run_websearch()
    
    # Example with custom query
    # run_websearch("What is the current status of Mars exploration?")