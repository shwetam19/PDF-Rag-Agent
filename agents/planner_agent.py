from agents import Agent
from config.settings import agent_model

# Import specialized agents for routing
from agents.rag_agent import rag_agent
from agents.summarization_agent import summarization_agent
from agents.specialized_agents import comparator_agent, timeline_agent, aggregator_agent

planner_agent = Agent(
    name="Planner",
    model=agent_model,
    instructions=(
        "You are the main orchestrator of the PDF Analysis System. "
        "Understand the user's intent and delegate to the correct specialist.\n\n"
        "ROUTING RULES:\n"
        "- SUMMARY -> Hand off to 'Summarization Agent'.\n"
        "- SPECIFIC QUESTION -> Hand off to 'RAG Agent'.\n"
        "- COMPARE -> Hand off to 'RAG Agent' first to find facts, then to 'Comparator Agent'.\n"
        "- TIMELINE -> Hand off to 'RAG Agent' first, then 'Timeline Builder'.\n"
        "- AGGREGATE -> Hand off to 'Aggregator Agent'.\n\n"
        "Do not answer directly. Always delegate."
    ),
    handoffs=[
        rag_agent,
        summarization_agent,
        comparator_agent,
        timeline_agent,
        aggregator_agent
    ]
)