from agents import Agent
from config.settings import agent_model

comparator_agent = Agent(
    name="Comparator Agent",
    model=agent_model,
    instructions="You are a Comparator Agent. Analyze differences and similarities across documents. Identify contradictions and provide a structured comparison."
)

timeline_agent = Agent(
    name="Timeline Builder",
    model=agent_model,
    instructions="You are a Timeline Builder. Extract dates and events to organize them chronologically. Highlight causal relationships."
)

aggregator_agent = Agent(
    name="Aggregator Agent",
    model=agent_model,
    instructions="You are an Aggregator. Merge overlapping information from multiple sources, eliminate redundancy, and create a unified narrative."
)