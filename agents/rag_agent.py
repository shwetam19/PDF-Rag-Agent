"""
RAG Agent using OpenAI Agents SDK
"""
import os
from typing import Dict, Any
from agents import Agent, Runner
from utils.vector_store import VectorStore
from config.settings import Config


class RAGAgent:
    """RAG Agent with retrieval"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.name = "RAG Agent"
        
        # Set OpenAI API key
        os.environ["OPENAI_API_KEY"] = Config.OPENAI_API_KEY
        
        # Define retrieval tool
        def retrieve_documents(query: str, top_k: int = 5):
            """Retrieve relevant document chunks based on query"""
            print(f"\n{'='*60}")
            print(f"RAG Agent: Retrieving for query: {query}")
            print(f"{'='*60}")
            
            retrieved = self.vector_store.search(query=query, top_k=top_k)
            
            if not retrieved:
                return "No relevant information found"
            
            print(f"Retrieved {len(retrieved)} chunks")
            
            # Build context
            context_parts = []
            for i, chunk_data in enumerate(retrieved, 1):
                chunk = chunk_data["chunk"]
                score = chunk_data["score"]
                context_parts.append(
                    f"[Source {i}] Document: {chunk.doc_name}, Page: {chunk.page_num}, "
                    f"Relevance: {score:.3f}\n{chunk.text}"
                )
            
            return "\n\n---\n\n".join(context_parts)
        
        # Create agent with tool
        self.agent = Agent(
            name="RAG Agent",
            instructions="""You are a RAG (Retrieval-Augmented Generation) Agent.

Your responsibilities:
1. Answer questions based ONLY on retrieved context
2. Always call retrieve_documents() first to get context
3. Reference specific documents and page numbers in your answers
4. If information is not in context, clearly state that
5. Cite sources using format: [Document Name, Page X]

Guidelines:
- Ground all answers in the retrieved context
- Do not make up or hallucinate information
- Be precise and factual
- Use direct quotes when appropriate
- Acknowledge limitations when context is insufficient""",
            tools=[retrieve_documents]
        )
        
        print(f"✓ {self.name} initialized")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process query using RAG"""
        query = input_data.get("query", "")
        
        if not query:
            return {
                "success": False,
                "data": None,
                "error": "No query provided"
            }
        
        try:
            # Run agent synchronously
            result = Runner.run_sync(self.agent, query)
            
            # Get evidence
            retrieved = self.vector_store.search(query=query, top_k=Config.TOP_K_RETRIEVAL)
            evidence = []
            
            for chunk_data in retrieved:
                evidence.append({
                    "doc_name": chunk_data["doc_name"],
                    "page_num": chunk_data["page_num"],
                    "chunk_id": chunk_data["chunk_id"],
                    "score": chunk_data["score"],
                    "text": chunk_data["chunk"].text,
                    "excerpt": chunk_data["excerpt"]
                })
            
            return {
                "success": True,
                "data": {
                    "answer": result.final_output,
                    "evidence": evidence,
                    "query": query
                },
                "metadata": {"retrieved_chunks": len(evidence)}
            }
        
        except Exception as e:
            print(f"Error in RAG Agent: {e}")
            return {
                "success": False,
                "data": None,
                "error": str(e)
            }