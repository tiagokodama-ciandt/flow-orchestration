"""
Tavily API integration for web search capabilities.
"""
import os
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults

load_dotenv()

# Initialize the Tavily search tool
search_tool = TavilySearchResults(
    api_key=os.getenv("TAVILY_API_KEY"),
    max_results=5,
    include_raw_content=False,
    include_images=False,
)
