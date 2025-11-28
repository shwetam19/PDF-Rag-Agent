"""
PDF Processing Utilities
"""
import pdfplumber
import fitz
from typing import List, Dict, Any
from dataclasses import dataclass
from config.settings import Config


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document"""
    doc_name: str
    page_num: int
    chunk_id: int
    text: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert chunk to dictionary"""
        return {
            "doc_name": self.doc_name,
            "page_num": self.page_num,
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata
        }


class PDFProcessor:
    """Handles PDF text extraction and chunking"""
    
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP
        
    def extract_text_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text from PDF with page-level granularity"""
        pages_data = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    text = page.extract_text()
                    if text:
                        pages_data.append({
                            "page_num": page_num,
                            "text": text.strip()
                        })
        except Exception as e:
            print(f"pdfplumber failed: {e}, trying PyMuPDF")
            try:
                doc = fitz.open(pdf_path)
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    text = page.get_text()
                    if text:
                        pages_data.append({
                            "page_num": page_num + 1,
                            "text": text.strip()
                        })
                doc.close()
            except Exception as e2:
                print(f"PyMuPDF also failed: {e2}")
                return []
        
        return pages_data
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start += self.chunk_size - self.chunk_overlap
        
        return chunks
    
    def process_pdf(self, pdf_path: str, doc_name: str) -> List[DocumentChunk]:
        """Process a PDF into chunks"""
        pages_data = self.extract_text_from_pdf(pdf_path)
        
        if not pages_data:
            return []
        
        chunks = []
        chunk_id = 0
        
        for page_data in pages_data:
            page_num = page_data["page_num"]
            page_text = page_data["text"]
            
            page_chunks = self.chunk_text(page_text)
            
            for chunk_text in page_chunks:
                chunk = DocumentChunk(
                    doc_name=doc_name,
                    page_num=page_num,
                    chunk_id=chunk_id,
                    text=chunk_text,
                    metadata={
                        "source": pdf_path,
                        "total_pages": len(pages_data)
                    }
                )
                chunks.append(chunk)
                chunk_id += 1
        
        return chunks
    
    def process_multiple_pdfs(self, pdf_files: List[tuple]) -> List[DocumentChunk]:
        """Process multiple PDFs"""
        all_chunks = []
        
        for pdf_path, doc_name in pdf_files:
            print(f"Processing: {doc_name}")
            chunks = self.process_pdf(pdf_path, doc_name)
            all_chunks.extend(chunks)
            print(f"  Created {len(chunks)} chunks")
        
        return all_chunks