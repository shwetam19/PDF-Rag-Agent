"""
Specialized Reasoning Agents using OpenAI Agents SDK
"""
import os
from typing import Dict, Any
from agents import Agent, Runner
from config.settings import Config


class ComparatorAgent:
    """Cross-document comparison agent"""
    
    def __init__(self):
        self.name = "Comparator Agent"
        os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
        
        self.agent = Agent(
            name="Comparator Agent",
            instructions="""You are a Comparator Agent specialized in analyzing differences and similarities.

Your responsibilities:
1. Identify key similarities across documents
2. Highlight important differences
3. Detect contradictions or conflicts
4. Provide structured comparative analysis
5. Use specific examples from each document

Guidelines:
- Be objective and balanced
- Use clear comparison frameworks
- Cite specific evidence
- Organize comparisons logically
- Highlight most significant differences"""
        )
        
        print(f"✓ {self.name} initialized")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare information"""
        query = input_data.get("query", "")
        evidence = input_data.get("evidence", [])
        
        if not evidence:
            return {"success": False, "data": None, "error": "No evidence"}
        
        evidence_text = "\n\n".join([
            f"[Source {i+1}] {ev['doc_name']} (Page {ev['page_num']})\n{ev.get('text', ev.get('excerpt', ''))}"
            for i, ev in enumerate(evidence)
        ])
        
        prompt = f"Query: {query}\n\nCompare and contrast the following information:\n\n{evidence_text}"
        
        try:
            result = Runner.run_sync(self.agent, prompt)
            return {
                "success": True,
                "data": {"comparison": result.final_output, "query": query},
                "metadata": {}
            }
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}


class TimelineBuilderAgent:
    """Chronological organization agent"""
    
    def __init__(self):
        self.name = "Timeline Builder Agent"
        os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
        
        self.agent = Agent(
            name="Timeline Builder Agent",
            instructions="""You are a Timeline Builder Agent specialized in chronological organization.

Your responsibilities:
1. Identify temporal markers (dates, times, sequences)
2. Organize events in chronological order
3. Establish causal relationships
4. Create clear temporal narratives
5. Handle relative time references

Guidelines:
- Extract all temporal information
- Order events logically
- Note simultaneity when relevant
- Highlight cause-and-effect relationships
- Use clear timeline format"""
        )
        
        print(f"✓ {self.name} initialized")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Build timeline"""
        query = input_data.get("query", "")
        evidence = input_data.get("evidence", [])
        
        if not evidence:
            return {"success": False, "data": None, "error": "No evidence"}
        
        evidence_text = "\n\n".join([
            f"[Event {i+1}] From {ev['doc_name']} (Page {ev['page_num']})\n{ev.get('text', ev.get('excerpt', ''))}"
            for i, ev in enumerate(evidence)
        ])
        
        prompt = f"Query: {query}\n\nConstruct a chronological timeline from these events:\n\n{evidence_text}"
        
        try:
            result = Runner.run_sync(self.agent, prompt)
            return {
                "success": True,
                "data": {"timeline": result.final_output, "query": query},
                "metadata": {}
            }
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}


class AggregatorAgent:
    """Information synthesis agent"""
    
    def __init__(self):
        self.name = "Aggregator Agent"
        os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
        
        self.agent = Agent(
            name="Aggregator Agent",
            instructions="""You are an Aggregator Agent specialized in information synthesis.

Your responsibilities:
1. Merge overlapping information
2. Eliminate redundancy
3. Preserve unique contributions from each source
4. Create comprehensive unified view
5. Maintain factual accuracy

Guidelines:
- Identify common themes
- Note unique perspectives
- Resolve minor contradictions
- Build complete picture
- Credit sources appropriately"""
        )
        
        print(f"✓ {self.name} initialized")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate information"""
        query = input_data.get("query", "")
        evidence = input_data.get("evidence", [])
        
        if not evidence:
            return {"success": False, "data": None, "error": "No evidence"}
        
        evidence_text = "\n\n".join([
            f"[Source {i+1}] {ev['doc_name']} (Page {ev['page_num']})\n{ev.get('text', ev.get('excerpt', ''))}"
            for i, ev in enumerate(evidence)
        ])
        
        prompt = f"Query: {query}\n\nAggregate and synthesize information from these sources:\n\n{evidence_text}"
        
        try:
            result = Runner.run_sync(self.agent, prompt)
            return {
                "success": True,
                "data": {"aggregation": result.final_output, "query": query},
                "metadata": {}
            }
        except Exception as e:
            return {"success": False, "data": None, "error": str(e)}