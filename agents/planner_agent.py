"""
Planner Agent - Orchestrator
"""
from typing import Dict, Any
from enum import Enum
from openai import OpenAI
from config.settings import Config


class IntentType(Enum):
    QUERY = "query"
    SUMMARIZE = "summarize"
    COMPARE = "compare"
    TIMELINE = "timeline"
    AGGREGATE = "aggregate"


class PlannerAgent:
    """Planner Agent for routing"""
    
    def __init__(self, rag_agent, summarization_agent, comparator_agent, timeline_agent, aggregator_agent):
        self.rag_agent = rag_agent
        self.summarization_agent = summarization_agent
        self.comparator_agent = comparator_agent
        self.timeline_agent = timeline_agent
        self.aggregator_agent = aggregator_agent
        
        self.client = OpenAI(api_key=Config.OPENAI_API_KEY)
        print("✓ Planner Agent initialized")
    
    def _detect_intent(self, user_input: str) -> IntentType:
        """Detect user intent"""
        response = self.client.chat.completions.create(
            model=Config.OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": """Classify into ONE category:
- QUERY: Questions about documents
- SUMMARIZE: Wants summary
- COMPARE: Wants comparison
- TIMELINE: Wants chronological order
- AGGREGATE: Wants synthesized info

Respond with ONLY the category name."""
                },
                {"role": "user", "content": user_input}
            ],
            temperature=0.3,
            max_tokens=10
        )
        
        intent_str = response.choices[0].message.content.strip().upper()
        intent_map = {
            "QUERY": IntentType.QUERY,
            "SUMMARIZE": IntentType.SUMMARIZE,
            "COMPARE": IntentType.COMPARE,
            "TIMELINE": IntentType.TIMELINE,
            "AGGREGATE": IntentType.AGGREGATE
        }
        return intent_map.get(intent_str, IntentType.QUERY)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Route to appropriate agent"""
        user_input = input_data.get("user_input", "")
        
        if not user_input:
            return {"success": False, "data": None, "error": "No input"}
        
        intent = self._detect_intent(user_input)
        trace = []
        
        print(f"\n📋 Planner detected: {intent.value}")
        
        if intent == IntentType.SUMMARIZE:
            trace.append("Planner → Summarization Agent")
            result = self.summarization_agent.process({})
            
            if result["success"]:
                return {
                    "success": True,
                    "data": {
                        "response": {"type": "summary", "content": result["data"]["summary"], "evidence": []},
                        "detected_intent": intent.value,
                        "execution_trace": trace
                    }
                }
        
        elif intent == IntentType.QUERY:
            trace.append("Planner → RAG Agent")
            result = self.rag_agent.process({"query": user_input})
            
            if result["success"]:
                return {
                    "success": True,
                    "data": {
                        "response": {"type": "query", "content": result["data"]["answer"], "evidence": result["data"]["evidence"]},
                        "detected_intent": intent.value,
                        "execution_trace": trace
                    }
                }
        
        elif intent == IntentType.COMPARE:
            trace.append("Planner → RAG → Comparator")
            rag_result = self.rag_agent.process({"query": user_input})
            
            if rag_result["success"]:
                compare_result = self.comparator_agent.process({
                    "query": user_input,
                    "evidence": rag_result["data"]["evidence"]
                })
                
                return {
                    "success": True,
                    "data": {
                        "response": {"type": "compare", "content": compare_result["data"]["comparison"], "evidence": rag_result["data"]["evidence"]},
                        "detected_intent": intent.value,
                        "execution_trace": trace
                    }
                }
        
        elif intent == IntentType.TIMELINE:
            trace.append("Planner → RAG → Timeline")
            rag_result = self.rag_agent.process({"query": user_input})
            
            if rag_result["success"]:
                timeline_result = self.timeline_agent.process({
                    "query": user_input,
                    "evidence": rag_result["data"]["evidence"]
                })
                
                return {
                    "success": True,
                    "data": {
                        "response": {"type": "timeline", "content": timeline_result["data"]["timeline"], "evidence": rag_result["data"]["evidence"]},
                        "detected_intent": intent.value,
                        "execution_trace": trace
                    }
                }
        
        elif intent == IntentType.AGGREGATE:
            trace.append("Planner → RAG → Aggregator")
            rag_result = self.rag_agent.process({"query": user_input})
            
            if rag_result["success"]:
                agg_result = self.aggregator_agent.process({
                    "query": user_input,
                    "evidence": rag_result["data"]["evidence"]
                })
                
                return {
                    "success": True,
                    "data": {
                        "response": {"type": "aggregate", "content": agg_result["data"]["aggregation"], "evidence": rag_result["data"]["evidence"]},
                        "detected_intent": intent.value,
                        "execution_trace": trace
                    }
                }
        
        return {"success": False, "data": None, "error": "Unknown intent"}