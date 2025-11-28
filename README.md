# 📚 Multi-Agent PDF Analysis System

Advanced document analysis system using OpenAI Agents SDK with autonomous multi-agent orchestration, RAG pipeline, and interactive UI.

## 🎯 Overview

This system implements a sophisticated multi-agent architecture using OpenAI Agents SDK (v0.6.1) for intelligent PDF document analysis. It features autonomous intent detection, retrieval-augmented generation (RAG), specialized reasoning agents, and an interactive Streamlit interface with citation highlighting.

## 🏗️ System Architecture

### Multi-Agent Framework

```
User Query → Planner Agent (Intent Detection)
              ↓
        Appropriate Agent Chain
              ↓
    RAG Agent (Retrieval + Generation)
              ↓
    Specialized Reasoning Agent
              ↓
    Response with Cited Evidence
```

### 6 Specialized Agents

- **Planner Agent** - Autonomous orchestrator using handoffs
- **RAG Agent** - Retrieval-augmented generation with FAISS
- **Summarization Agent** - Full-document summarization
- **Comparator Agent** - Cross-document comparison analysis
- **Timeline Builder Agent** - Chronological event organization
- **Aggregator Agent** - Multi-source information synthesis

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| Agent Framework | OpenAI Agents SDK v0.6.1 |
| LLM | OpenAI (provider-agnostic) |
| Vector Database | FAISS (IndexFlatIP) |
| Embeddings | sentence-transformers (384-dim) |
| PDF Processing | pdfplumber + PyMuPDF |
| UI Framework | Streamlit |

## 📋 Features

### Core Capabilities

✅ Autonomous Intent Detection - No manual mode selection  
✅ RAG Pipeline - Semantic search with grounded responses  
✅ Multi-Document Analysis - Cross-document retrieval  
✅ Citation Tracking - Every answer includes ranked evidence  
✅ Interactive PDF Viewer - Click-to-navigate with highlighting  
✅ Agent Orchestration - Dynamic agent chaining  

### Advanced Features

✅ Tool Calling - Agents call Python functions (@function_tool)  
✅ Autonomous Handoffs - LLM-driven delegation (no manual routing)  
✅ Global State Management - Tools access shared Vector Store  
✅ Evidence Highlighting - Yellow highlights on cited passages  
✅ Execution Tracing - Transparent agent workflow via Runner logs  

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.9+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys)) OR a Gemini API Key

### 2. Installation

```bash
# Clone repository
git clone <repository-url>
cd pdf_agent_system

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your API key
OPENAI_API_KEY=your_key_here
```

### 4. Run Application

```bash
streamlit run app.py
```

Access the application at: http://localhost:8501

## 📁 Project Structure

```
pdf_agent_system/
├── agents/
│   ├── __init__.py
│   ├── tools.py                    # Standalone tools for SDK agents
│   ├── rag_agent.py                # RAG Agent definition
│   ├── summarization_agent.py      # Summarization Agent definition
│   ├── specialized_agents.py       # Reasoning Agents (Comparator, Timeline, etc.)
│   └── planner_agent.py            # Orchestrator with Handoffs
├── utils/
│   ├── __init__.py
│   ├── state.py                    # Singleton for tool access
│   ├── pdf_processor.py            # PDF extraction + chunking
│   └── vector_store.py             # FAISS vector database
├── config/
│   ├── __init__.py
│   └── settings.py                 # Configuration
├── app.py                          # Streamlit UI
├── requirements.txt                # Dependencies
├── .env.example                    # Configuration template
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

## 🎓 How It Works

### 1. OpenAI Agents SDK Integration

We use the native Agent and Runner primitives:

```python
from agents import Agent, Runner

# Agents invoke tools and hand off to others
result = Runner.run_sync(planner_agent, user_query)
print(result.final_output)
```

### 2. Tool Functions

Tools are defined using the @function_tool decorator and access shared state:

```python
@function_tool
def retrieve_documents(query: str):
    """Retrieve relevant chunks"""
    return global_state.vector_store.search(query)
```

### 3. Autonomous Orchestration

The Planner Agent uses instructions and the handoffs list to route dynamically:

```python
planner_agent = Agent(
    name="Planner",
    instructions="Route queries to the correct specialist...",
    handoffs=[rag_agent, summarization_agent, comparator_agent]
)
```

## 💡 Usage Examples

### Example 1: Question Answering

**User:** "What are the main findings in the research paper?"

**System Flow:**
1. Planner delegates to RAG Agent
2. RAG Agent calls 'retrieve_documents' tool
3. Agent generates answer with citations

**Output:**
```
Answer: "The research identifies three main findings: [1] X, [2] Y, [3] Z"
```

### Example 2: Comparative Analysis

**User:** "Compare the methodologies across these papers"

**System Flow:**
1. Planner delegates to RAG Agent
2. RAG Agent retrieves methodology sections
3. RAG Agent hands off to Comparator Agent
4. Comparator Agent analyzes differences

**Output:**
```
Structured comparison with specific examples
```

## ⚙️ Configuration

### Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional (defaults shown)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
```

## 💰 Cost Considerations

### OpenAI Pricing

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| gpt-4o-mini | $0.150 | $0.600 |
| gpt-4o | $2.50 | $10.00 |

### Typical Usage

- **Per Query:** ~2,000 input + 500 output tokens = ~$0.0006
- **Per Session:** ~10 queries = ~$0.006

## 🎉 Acknowledgments

- **OpenAI** - Agents SDK framework
- **Facebook Research** - FAISS vector search
- **Sentence Transformers** - Embedding models
- **Streamlit** - Interactive UI framework

---

✨ **Built with OpenAI Agents SDK v0.6.1 | Multi-Agent Architecture** ✨