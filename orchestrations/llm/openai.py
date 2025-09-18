import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

model = init_chat_model(
    base_url="https://flow.ciandt.com/ai-orchestration-api/v1/openai",
    model="gpt-4o-mini",
    default_headers={"FlowAgent": "Flow Api"},
    api_key=os.environ["OPENAI_API_KEY"],
)
