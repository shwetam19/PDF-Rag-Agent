from agents import Agent
from config.settings import agent_model
from agents.tools import retrieve_documents
from agents.specialized_agents import comparator_agent, timeline_agent, aggregator_agent

rag_agent = Agent(
    name="RAG Agent",
    model=agent_model,
    instructions=(
        "You are a RAG Specialist. Answer questions using ONLY the provided evidence.\n"
        "1. ALWAYS call 'retrieve_documents' first.\n"
        "2. Cite sources as [1], [2] corresponding to tool output.\n"
        "3. If complex reasoning is needed (comparison, timeline), you can hand off to a specialist."
    ),
    tools=[retrieve_documents],
    # Critical: RAG Agent can hand off to reasoning agents to complete the chain
    handoffs=[comparator_agent, timeline_agent, aggregator_agent]
)