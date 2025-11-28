from agents import Agent
from config.settings import agent_model
from agents.tools import get_full_document_content

summarization_agent = Agent(
    name="Summarization Agent",
    model=agent_model,
    instructions=(
        "You are an expert Document Summarizer. "
        "Your goal is to provide a comprehensive, abstractive summary of the available documents.\n"
        "1. Call 'get_full_document_content' to read the documents.\n"
        "2. Synthesize key themes, findings, and conclusions into a coherent narrative."
    ),
    tools=[get_full_document_content]
)