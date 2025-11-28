"""
Vector Store for Document Embeddings
"""
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any
from utils.pdf_processor import DocumentChunk
from config.settings import Config


class VectorStore:
    """FAISS vector database for document retrieval"""
    
    def __init__(self, embedding_model_name: str = None):
        model_name = embedding_model_name or Config.EMBEDDING_MODEL
        self.embedding_model = SentenceTransformer(model_name)
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        self.index = None
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray = None
        
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """Create embeddings for texts"""
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings
    
    def build_index(self, chunks: List[DocumentChunk]):
        """Build FAISS index from chunks"""
        if not chunks:
            raise ValueError("No chunks provided")
        
        self.chunks = chunks
        texts = [chunk.text for chunk in chunks]
        
        print(f"Creating embeddings for {len(texts)} chunks...")
        self.embeddings = self.create_embeddings(texts)
        
        print("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(self.embeddings)
        
        print(f"✓ Index built with {self.index.ntotal} vectors")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query_embedding = self.create_embeddings([query])
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                results.append({
                    "chunk": chunk,
                    "score": float(score),
                    "doc_name": chunk.doc_name,
                    "page_num": chunk.page_num,
                    "chunk_id": chunk.chunk_id,
                    "excerpt": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                })
        
        return results
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about stored documents"""
        stats = {}
        
        for chunk in self.chunks:
            doc_name = chunk.doc_name
            if doc_name not in stats:
                stats[doc_name] = {
                    "chunk_count": 0,
                    "page_count": 0,
                    "total_chars": 0
                }
            
            stats[doc_name]["chunk_count"] += 1
            stats[doc_name]["page_count"] = max(stats[doc_name]["page_count"], chunk.page_num)
            stats[doc_name]["total_chars"] += len(chunk.text)
        
        return stats