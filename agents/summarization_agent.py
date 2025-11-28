"""
Summarization Agent using OpenAI Agents SDK
"""
import os
from typing import Dict, Any
from agents import Agent, Runner
from utils.vector_store import VectorStore
from config.settings import Config


class SummarizationAgent:
    """Summarization Agent with map-reduce"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.name = "Summarization Agent"
        
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
        
        # Create agent
        self.agent = Agent(
            name="Summarization Agent",
            instructions="""You are a Summarization Agent specialized in creating comprehensive document summaries.

Your responsibilities:
1. Extract key themes and main points
2. Maintain factual accuracy
3. Avoid unnecessary details while preserving context
4. Create coherent narratives
5. Identify common themes across multiple documents

Guidelines:
- Focus on substance over style
- Preserve critical information
- Remove redundancy
- Highlight important findings
- Maintain logical flow"""
        )
        
        print(f"✓ {self.name} initialized")
    
    def _summarize_batch(self, chunks: list) -> str:
        """Summarize a batch of chunks"""
        combined = "\n\n".join(chunks)
        prompt = f"Summarize the following text, preserving key information:\n\n{combined}"
        
        try:
            result = Runner.run_sync(self.agent, prompt)
            return result.final_output
        except Exception as e:
            print(f"Error summarizing batch: {e}")
            return "Error generating summary"
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary"""
        all_chunks = self.vector_store.chunks
        texts = [chunk.text for chunk in all_chunks]
        
        if not texts:
            return {"success": False, "data": None, "error": "No documents"}
        
        print(f"\n{'='*60}")
        print(f"Summarization Agent: Processing {len(texts)} chunks")
        print(f"{'='*60}")
        
        try:
            # Map-reduce for long documents
            if len(texts) > 10:
                batch_size = 10
                summaries = []
                
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i + batch_size]
                    batch_summary = self._summarize_batch(batch)
                    summaries.append(batch_summary)
                
                # Reduce
                if len(summaries) > 1:
                    final_summary = self._summarize_batch(summaries)
                else:
                    final_summary = summaries[0]
            else:
                final_summary = self._summarize_batch(texts)
            
            return {
                "success": True,
                "data": {
                    "summary": final_summary,
                    "chunks_processed": len(texts)
                },
                "metadata": {"total_chunks": len(texts)}
            }
        
        except Exception as e:
            print(f"Error in Summarization Agent: {e}")
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }