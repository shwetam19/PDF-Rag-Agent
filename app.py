"""
Multi-Agent PDF Analysis System - Streamlit Application
"""
import streamlit as st
import os
import tempfile
from pathlib import Path
import fitz  # PyMuPDF for PDF viewing
from PIL import Image
import io

from config.settings import Config
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStore
from agents.rag_agent import RAGAgent
from agents.summarization_agent import SummarizationAgent
from agents.specialized_agents import ComparatorAgent, TimelineBuilderAgent, AggregatorAgent
from agents.planner_agent import PlannerAgent


# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide"
)


def initialize_session_state():
    """Initialize Streamlit session state variables"""
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.vector_store = None
        st.session_state.planner = None
        st.session_state.chat_history = []
        st.session_state.uploaded_files = []
        st.session_state.pdf_documents = {}  # Store PDF file paths
        st.session_state.current_page = {}  # Track current page for each document
        st.session_state.highlight_text = {}  # Track text to highlight for each document


def save_uploaded_files(uploaded_files):
    """
    Save uploaded files to temporary directory.
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects
    
    Returns:
        List of tuples (file_path, doc_name)
    """
    temp_dir = tempfile.mkdtemp()
    saved_files = []
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files.append((file_path, uploaded_file.name))
        st.session_state.pdf_documents[uploaded_file.name] = file_path
    
    return saved_files


def initialize_system(pdf_files):
    """
    Initialize the multi-agent system with uploaded PDFs.
    
    Args:
        pdf_files: List of tuples (file_path, doc_name)
    """
    with st.spinner("🔧 Initializing system..."):
        # Process PDFs
        st.info("📄 Processing PDF documents...")
        pdf_processor = PDFProcessor()
        chunks = pdf_processor.process_multiple_pdfs(pdf_files)
        
        if not chunks:
            st.error("Failed to process PDF documents. Please check the files.")
            return False
        
        # Build vector store
        st.info("🔍 Building vector index...")
        vector_store = VectorStore()
        vector_store.build_index(chunks)
        st.session_state.vector_store = vector_store
        
        # Initialize agents
        st.info("🤖 Initializing agents...")
        rag_agent = RAGAgent(vector_store)
        summarization_agent = SummarizationAgent(vector_store)
        comparator_agent = ComparatorAgent()
        timeline_agent = TimelineBuilderAgent()
        aggregator_agent = AggregatorAgent()
        
        # Initialize planner
        planner = PlannerAgent(
            rag_agent=rag_agent,
            summarization_agent=summarization_agent,
            comparator_agent=comparator_agent,
            timeline_agent=timeline_agent,
            aggregator_agent=aggregator_agent
        )
        st.session_state.planner = planner
        st.session_state.initialized = True
        
        # Show document stats
        stats = vector_store.get_document_stats()
        st.success(f"✅ System initialized! Processed {len(stats)} document(s) with {len(chunks)} chunks total.")
        
        return True


def render_pdf_page(pdf_path, page_num, highlight_text=None):
    """
    Render a specific page from a PDF with optional text highlighting.
    
    Args:
        pdf_path: Path to PDF file
        page_num: Page number (1-indexed)
        highlight_text: Text to highlight on the page (optional)
    
    Returns:
        PIL Image of the page
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]  # Convert to 0-indexed
        
        # Highlight text if provided
        if highlight_text and len(highlight_text.strip()) > 10:
            # Get all text from page for debugging
            page_text = page.get_text()
            
            # Normalize both texts for comparison
            def normalize(text):
                # Remove extra whitespace, newlines, tabs
                text = ' '.join(text.split())
                # Remove common punctuation issues
                text = text.replace('\n', ' ').replace('\r', ' ')
                return text.strip()
            
            normalized_highlight = normalize(highlight_text)
            normalized_page = normalize(page_text)
            
            # Try different search strategies
            search_attempts = [
                normalized_highlight[:100],  # First 100 chars
                normalized_highlight[:80],   # First 80 chars
                normalized_highlight[:60],   # First 60 chars
                normalized_highlight.split('.')[0][:50] if '.' in normalized_highlight else normalized_highlight[:50],  # First sentence
                ' '.join(normalized_highlight.split()[:10]),  # First 10 words
                ' '.join(normalized_highlight.split()[:7]),   # First 7 words
                ' '.join(normalized_highlight.split()[:5]),   # First 5 words
            ]
            
            text_instances = []
            for search_text in search_attempts:
                if len(search_text) < 10:  # Skip too short searches
                    continue
                    
                # Try to find in page
                instances = page.search_for(search_text)
                if instances:
                    text_instances = instances
                    print(f"✓ Found match with: {search_text[:50]}...")
                    break
            
            # If still no match, try word-by-word search for key phrases
            if not text_instances:
                words = normalized_highlight.split()
                # Try 3-word phrases sliding window
                for i in range(len(words) - 2):
                    phrase = ' '.join(words[i:i+3])
                    instances = page.search_for(phrase)
                    if instances:
                        text_instances = instances
                        print(f"✓ Found 3-word match: {phrase}")
                        break
            
            # Highlight all found instances
            if text_instances:
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=(1, 1, 0))  # Yellow highlight
                    highlight.update()
                print(f"✓ Highlighted {len(text_instances)} instance(s)")
            else:
                print(f"✗ Could not find text to highlight on page {page_num}")
                print(f"  Looking for: {normalized_highlight[:100]}...")
        
        # Render page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        doc.close()
        return img
    except Exception as e:
        print(f"Error rendering page: {e}")
        import traceback
        traceback.print_exc()
        return None


def display_evidence_sidebar(evidence):
    """
    Display evidence citations in the sidebar.
    
    Args:
        evidence: List of evidence dictionaries
    """
    if not evidence:
        return
    
    st.sidebar.markdown("### 📚 Evidence Citations")
    
    for i, ev in enumerate(evidence, 1):
        with st.sidebar.expander(f"📌 Citation {i} (Score: {ev['score']:.3f})"):
            st.markdown(f"**Document:** {ev['doc_name']}")
            st.markdown(f"**Page:** {ev['page_num']}")
            st.markdown(f"**Chunk ID:** {ev['chunk_id']}")
            st.markdown(f"**Excerpt:**\n{ev['excerpt']}")
            
            # Button to navigate to this citation with highlighting
            if st.button(f"View in Document", key=f"nav_{i}"):
                st.session_state.current_page[ev['doc_name']] = ev['page_num']
                # Use full text from evidence
                highlight_text = ev.get('text', ev['excerpt'])
                st.session_state.highlight_text[ev['doc_name']] = highlight_text
                st.rerun()


def display_document_navigator():
    """Display interactive PDF document navigator with text highlighting"""
    if not st.session_state.pdf_documents:
        return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Document Navigator")
    
    # Document selector
    doc_names = list(st.session_state.pdf_documents.keys())
    selected_doc = st.sidebar.selectbox("Select Document", doc_names)
    
    if selected_doc:
        pdf_path = st.session_state.pdf_documents[selected_doc]
        
        # Get total pages
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            # Initialize current page if not exists
            if selected_doc not in st.session_state.current_page:
                st.session_state.current_page[selected_doc] = 1
            
            # Initialize highlight text if not exists
            if selected_doc not in st.session_state.highlight_text:
                st.session_state.highlight_text[selected_doc] = None
            
            # Page navigation
            col1, col2, col3 = st.sidebar.columns([1, 2, 1])
            
            with col1:
                if st.button("◀", key="prev_page"):
                    if st.session_state.current_page[selected_doc] > 1:
                        st.session_state.current_page[selected_doc] -= 1
                        st.rerun()
            
            with col2:
                current_page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=st.session_state.current_page[selected_doc],
                    key="page_input"
                )
                st.session_state.current_page[selected_doc] = current_page
            
            with col3:
                if st.button("▶", key="next_page"):
                    if st.session_state.current_page[selected_doc] < total_pages:
                        st.session_state.current_page[selected_doc] += 1
                        st.rerun()
            
            # Show if text is being highlighted
            if st.session_state.highlight_text[selected_doc]:
                st.sidebar.caption("🟡 Highlighting active")
                with st.sidebar.expander("Debug: Search Text"):
                    preview = st.session_state.highlight_text[selected_doc][:200]
                    st.text(f"{preview}...")
                    if st.button("Clear Highlight"):
                        st.session_state.highlight_text[selected_doc] = None
                        st.rerun()
            
            # Render the page with highlighting
            highlight = st.session_state.highlight_text[selected_doc]
            img = render_pdf_page(pdf_path, st.session_state.current_page[selected_doc], highlight)
            if img:
                st.sidebar.image(img, caption=f"{selected_doc} - Page {current_page}/{total_pages}", use_container_width=True)
        
        except Exception as e:
            st.sidebar.error(f"Error loading document: {e}")


def display_chat_interface():
    """Display the main chat interface"""
    st.title("💬 Multi-Agent PDF Analysis")
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display evidence if available WITH CLICKABLE CITATIONS
                if "evidence" in message and message["evidence"]:
                    with st.expander("📎 View Evidence"):
                        for i, ev in enumerate(message["evidence"], 1):
                            # Display citation info with all metadata
                            st.markdown(f"**{i}. {ev['doc_name']}** (Page {ev['page_num']}, Chunk #{ev['chunk_id']}, Score: {ev['score']:.3f})")
                            st.text(ev['excerpt'][:150] + "...")
                            
                            # CLICKABLE BUTTON TO NAVIGATE TO CITED PAGE WITH HIGHLIGHTING
                            if st.button(f"📍 Go to Page {ev['page_num']}", key=f"cite_{id(message)}_{i}"):
                                st.session_state.current_page[ev['doc_name']] = ev['page_num']
                                # Use the full text stored in evidence
                                highlight_text = ev.get('text', ev['excerpt'])
                                st.session_state.highlight_text[ev['doc_name']] = highlight_text
                                st.rerun()
                            st.markdown("---")
                
                # Display execution trace if available
                if "trace" in message and message["trace"]:
                    with st.expander("🔍 Execution Trace"):
                        for step in message["trace"]:
                            st.text(f"  → {step}")
    
    # Chat input
    user_input = st.chat_input("Ask a question or request a summary...")
    
    if user_input:
        # Add user message to chat
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Process with planner
        with st.spinner("🤔 Processing your request..."):
            result = st.session_state.planner.process({
                "user_input": user_input
            })
            
            if result["success"]:
                response_data = result["data"]["response"]
                content = response_data["content"]
                evidence = response_data.get("evidence", [])
                trace = result["data"].get("execution_trace", [])
                intent = result["data"].get("detected_intent", "unknown")
                
                # Add assistant response to chat
                assistant_message = {
                    "role": "assistant",
                    "content": content,
                    "evidence": evidence,
                    "trace": trace,
                    "intent": intent
                }
                st.session_state.chat_history.append(assistant_message)
                
                # Update sidebar with evidence
                if evidence:
                    display_evidence_sidebar(evidence)
            else:
                error_message = result.get("error", "Unknown error occurred")
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"❌ Error: {error_message}"
                })
        
        st.rerun()


def main():
    """Main application function"""
    initialize_session_state()
    
    # Main content
    if not st.session_state.initialized:
        # Upload section on main page
        st.title("💬 Multi-Agent PDF Analysis")
        st.markdown("---")
        
        # File uploader on main page
        uploaded_files = st.file_uploader(
            "📁 Upload PDF documents to begin",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            if st.button("🚀 Initialize System", type="primary"):
                saved_files = save_uploaded_files(uploaded_files)
                st.session_state.uploaded_files = [f[1] for f in saved_files]
                
                try:
                    Config.validate()
                    if initialize_system(saved_files):
                        st.rerun()
                except ValueError as e:
                    st.error(f"Configuration error: {e}")
                    st.info("Please set OPENAI_API_KEY in your environment variables or .env file")
    else:
        # Sidebar for document viewer and stats
        with st.sidebar:
            st.title("📖 Documents")
            
            # Document stats
            if st.session_state.vector_store:
                st.markdown("### 📊 Statistics")
                stats = st.session_state.vector_store.get_document_stats()
                
                for doc_name, doc_stats in stats.items():
                    with st.expander(f"📄 {doc_name}"):
                        st.write(f"**Chunks:** {doc_stats['chunk_count']}")
                        st.write(f"**Pages:** {doc_stats['page_count']}")
                        st.write(f"**Total Characters:** {doc_stats['total_chars']:,}")
            
            # Clear chat button
            if st.session_state.chat_history:
                st.markdown("---")
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        # Display chat interface
        col1, col2 = st.columns([3, 1])
        
        with col1:
            display_chat_interface()
        
        with col2:
            display_document_navigator()


if __name__ == "__main__":
    main()