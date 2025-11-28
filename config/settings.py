import os
from openai import AsyncOpenAI
from agents import OpenAIChatCompletionsModel
from config.settings import Config

# Ensure API Key is loaded
if not Config.OPENAI_API_KEY:
    raise ValueError("API Key not found. Please check your .env file.")

# Initialize OpenAI Client
client = AsyncOpenAI(
    api_key=Config.OPENAI_API_KEY,
    base_url="https://platform.openai.com/api-keys" 
)

# Create the Model Wrapper used by all agents
agent_model = OpenAIChatCompletionsModel(
    model=Config.OPENAI_MODEL, 
    openai_client=client,
)