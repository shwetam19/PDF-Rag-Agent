"""
Multi-Agent PDF Analysis System - Streamlit Application
"""
import streamlit as st
import os
import tempfile
import fitz 
from PIL import Image
import io

from agents import Runner
from config.settings import Config
from utils.pdf_processor import PDFProcessor
from utils.vector_store import VectorStore
from utils.state import global_state # Tools need this to access data
from agents.planner_agent import planner_agent # Import the pre-made agent instance

# Page configuration
st.set_page_config(
    page_title=Config.PAGE_TITLE,
    page_icon=Config.PAGE_ICON,
    layout="wide"
)

def initialize_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
        st.session_state.messages = []
        st.session_state.uploaded_files = []
        st.session_state.pdf_documents = {} 
        st.session_state.current_page = {}
        st.session_state.highlight_text = {}

def save_uploaded_files(uploaded_files):
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
    with st.spinner("🔧 Initializing system..."):
        st.info("📄 Processing PDF documents...")
        pdf_processor = PDFProcessor()
        chunks = pdf_processor.process_multiple_pdfs(pdf_files)
        
        if not chunks:
            st.error("Failed to process PDF documents.")
            return False
        
        st.info("🔍 Building vector index...")
        vector_store = VectorStore()
        vector_store.build_index(chunks)
        
        # Store in global state for Tools to access ---
        global_state.vector_store = vector_store
        
        st.session_state.initialized = True
        
        stats = vector_store.get_document_stats()
        st.success(f"✅ System initialized! Processed {len(stats)} document(s).")
        return True

def render_pdf_page(pdf_path, page_num, highlight_text=None):
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        
        if highlight_text and len(highlight_text.strip()) > 10:
            text_instances = page.search_for(highlight_text[:50])
            if text_instances:
                for inst in text_instances:
                    highlight = page.add_highlight_annot(inst)
                    highlight.set_colors(stroke=(1, 1, 0))
                    highlight.update()
        
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        doc.close()
        return img
    except Exception as e:
        return None

def display_chat_interface():
    st.title("💬 Multi-Agent PDF Analysis")
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "trace" in message and message["trace"]:
                with st.expander("🔍 Execution Trace"):
                    for step in message["trace"]:
                        st.text(step)

    if prompt := st.chat_input("Ask a question, request a summary..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_container = st.status("🤖 Agents thinking...", expanded=True)
            execution_trace = []
            
            try:
                # --- RUNNER EXECUTION (New SDK Syntax) ---
                result = Runner.run_sync(planner_agent, prompt)
                
                # Extract trace
                status_container.write("Execution Trace:")
                for msg in result.new_messages():
                    if msg.role == "assistant" and msg.content:
                        trace_step = f"Agent Action: {msg.content[:100]}..."
                        execution_trace.append(trace_step)
                        status_container.markdown(f"**Step:** {trace_step}")
                
                status_container.update(label="✅ Complete", state="complete")
                
                st.markdown(result.final_output)
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.final_output,
                    "trace": execution_trace
                })
                
            except Exception as e:
                status_container.update(label="❌ Error", state="error")
                st.error(f"Error: {str(e)}")

def display_document_navigator():
    if not st.session_state.pdf_documents: return
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Document Navigator")
    
    doc_names = list(st.session_state.pdf_documents.keys())
    selected_doc = st.sidebar.selectbox("Select Document", doc_names)
    
    if selected_doc:
        pdf_path = st.session_state.pdf_documents[selected_doc]
        if selected_doc not in st.session_state.current_page:
            st.session_state.current_page[selected_doc] = 1
            
        col1, col2, col3 = st.sidebar.columns([1, 2, 1])
        with col1:
            if st.button("◀", key="prev"):
                st.session_state.current_page[selected_doc] = max(1, st.session_state.current_page[selected_doc] - 1)
                st.rerun()
        with col3:
            if st.button("▶", key="next"):
                st.session_state.current_page[selected_doc] += 1
                st.rerun()
                
        img = render_pdf_page(pdf_path, st.session_state.current_page[selected_doc], st.session_state.highlight_text.get(selected_doc))
        if img: st.sidebar.image(img, use_container_width=True)

def main():
    initialize_session_state()
    
    if not st.session_state.initialized:
        st.title("📂 Upload PDFs")
        uploaded_files = st.file_uploader("Upload", type=["pdf"], accept_multiple_files=True)
        if uploaded_files and st.button("🚀 Initialize System", type="primary"):
            try:
                Config.validate()
                saved_files = save_uploaded_files(uploaded_files)
                initialize_system(saved_files)
                st.rerun()
            except ValueError as e:
                st.error(f"Configuration error: {e}")
    else:
        col1, col2 = st.columns([3, 1])
        with col1: display_chat_interface()
        with col2: display_document_navigator()

if __name__ == "__main__":
    main()