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

from agents import Runner
from config.settings import Config
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStore
from utils.state import global_state  # Import singleton state
from agents.planner_agent import planner_agent # Import the pre-configured planner

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
        st.session_state.chat_history = []
        st.session_state.uploaded_files = []
        st.session_state.pdf_documents = {}  # Store PDF file paths
        st.session_state.current_page = {}  # Track current page for each document
        st.session_state.highlight_text = {}  # Track text to highlight for each document


def save_uploaded_files(uploaded_files):
    """Save uploaded files to temporary directory."""
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
        
        # --- CRITICAL CHANGE for SDK Tools ---
        # Store the vector store in the global state so tools can access it
        global_state.vector_store = vector_store
        st.session_state.vector_store = vector_store
        
        st.session_state.initialized = True
        
        # Show document stats
        stats = vector_store.get_document_stats()
        st.success(f"✅ System initialized! Processed {len(stats)} document(s) with {len(chunks)} chunks total.")
        
        return True


def render_pdf_page(pdf_path, page_num, highlight_text=None):
    """Render a specific page from a PDF with optional text highlighting."""
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]  # Convert to 0-indexed
        
        # Highlight text if provided
        if highlight_text and len(highlight_text.strip()) > 10:
            text_instances = page.search_for(highlight_text[:50]) # Search for snippet
            
            if text_instances:
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=(1, 1, 0))  # Yellow highlight
                    highlight.update()
        
        # Render page to image
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
        img_data = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_data))
        
        doc.close()
        return img
    except Exception as e:
        st.error(f"Error rendering page: {e}")
        return None


def display_evidence_sidebar(evidence):
    """Display evidence citations in the sidebar."""
    if not evidence:
        return
    
    st.sidebar.markdown("### 📚 Evidence Citations")
    
    # In the new SDK structure, evidence extraction logic needs to parse the LLM output
    # or rely on tool outputs captured in the trace. 
    # For simplicity, we assume the LLM output contains formatted citations we can parse
    # or this function can be adapted to read from trace logs if implemented.
    pass 


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
        
        try:
            doc = fitz.open(pdf_path)
            total_pages = len(doc)
            doc.close()
            
            if selected_doc not in st.session_state.current_page:
                st.session_state.current_page[selected_doc] = 1
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
                st.write(f"Page {st.session_state.current_page[selected_doc]} of {total_pages}")
            with col3:
                if st.button("▶", key="next_page"):
                    if st.session_state.current_page[selected_doc] < total_pages:
                        st.session_state.current_page[selected_doc] += 1
                        st.rerun()
            
            # Render page
            highlight = st.session_state.highlight_text[selected_doc]
            img = render_pdf_page(pdf_path, st.session_state.current_page[selected_doc], highlight)
            if img:
                st.sidebar.image(img, use_container_width=True)
        
        except Exception as e:
            st.sidebar.error(f"Error loading document: {e}")


def display_chat_interface():
    """Display the main chat interface"""
    st.title("💬 Multi-Agent PDF Analysis")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display trace if available
            if "trace" in message and message["trace"]:
                with st.expander("🔍 Execution Trace"):
                    for step in message["trace"]:
                        st.text(f"→ {step}")

    # Chat input
    if prompt := st.chat_input("Ask a question, request a summary..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Process with Agent Runner
        with st.chat_message("assistant"):
            status_container = st.status("🤖 Agents thinking...", expanded=True)
            execution_trace = []
            
            try:
                # --- RUNNER EXECUTION (New SDK Syntax) ---
                # Runner.run_sync handles the Planner -> Handoff -> Tool -> Output loop
                result = Runner.run_sync(planner_agent, prompt)
                
                # Extract reasoning trace from new messages
                status_container.write("Execution Trace:")
                for msg in result.new_messages():
                    if msg.role == "assistant" and msg.content:
                        # Simple logic to capture agent thoughts/handoffs
                        trace_step = f"Agent Action: {msg.content[:100]}..."
                        execution_trace.append(trace_step)
                        status_container.markdown(f"**Step:** {trace_step}")
                
                status_container.update(label="✅ Complete", state="complete")
                
                # Show Final Output
                final_response = result.final_output
                st.markdown(final_response)
                
                # Save to history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": final_response,
                    "trace": execution_trace
                })
                
            except Exception as e:
                status_container.update(label="❌ Error", state="error")
                st.error(f"Error: {str(e)}")


def main():
    """Main application function"""
    initialize_session_state()
    
    if not st.session_state.initialized:
        st.title("📂 Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload PDF documents to begin",
            type=["pdf"],
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        
        if uploaded_files:
            if st.button("🚀 Initialize System", type="primary"):
                try:
                    Config.validate()
                    saved_files = save_uploaded_files(uploaded_files)
                    if initialize_system(saved_files):
                        st.rerun()
                except ValueError as e:
                    st.error(f"Configuration error: {e}")
                    st.info("Please set OPENAI_API_KEY in your environment variables or .env file")
    else:
        # Sidebar for document viewer and stats
        with st.sidebar:
            st.title("📖 Documents")
            if st.session_state.vector_store:
                st.markdown("### 📊 Statistics")
                stats = st.session_state.vector_store.get_document_stats()
                for doc_name, doc_stats in stats.items():
                    with st.expander(f"📄 {doc_name}"):
                        st.write(f"Chunks: {doc_stats['chunk_count']}")
                        st.write(f"Pages: {doc_stats['page_count']}")
            
            if st.session_state.chat_history:
                st.markdown("---")
                if st.button("🗑️ Clear Chat History"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        col1, col2 = st.columns([3, 1])
        with col1:
            display_chat_interface()
        with col2:
            display_document_navigator()


if __name__ == "__main__":
    main()