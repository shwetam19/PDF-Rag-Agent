# 📚 Multi-Agent PDF Analysis System

**Advanced document analysis system using OpenAI Agents SDK with autonomous multi-agent orchestration, RAG pipeline, and interactive UI.**

---

## 🎯 Overview

This system implements a sophisticated multi-agent architecture using **OpenAI Agents SDK (v0.6.1)** for intelligent PDF document analysis. It features autonomous intent detection, retrieval-augmented generation (RAG), specialized reasoning agents, and an interactive Streamlit interface with citation highlighting.

---

## 🏗️ System Architecture

### **Multi-Agent Framework**

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

### **6 Specialized Agents**

1. **Planner Agent** - LLM-based intent classification and routing
2. **RAG Agent** - Retrieval-augmented generation with FAISS
3. **Summarization Agent** - Map-reduce document summarization
4. **Comparator Agent** - Cross-document comparison analysis
5. **Timeline Builder Agent** - Chronological event organization
6. **Aggregator Agent** - Multi-source information synthesis

---

## 🔧 Technical Stack

| Component | Technology |
|-----------|-----------|
| **Agent Framework** | OpenAI Agents SDK v0.6.1 |
| **LLM** | OpenAI (provider-agnostic) |
| **Vector Database** | FAISS (IndexFlatIP) |
| **Embeddings** | sentence-transformers (384-dim) |
| **PDF Processing** | pdfplumber + PyMuPDF |
| **UI Framework** | Streamlit |

---

## 📋 Features

### **Core Capabilities**
- ✅ **Autonomous Intent Detection** - No manual mode selection
- ✅ **RAG Pipeline** - Semantic search with grounded responses
- ✅ **Multi-Document Analysis** - Cross-document retrieval
- ✅ **Citation Tracking** - Every answer includes ranked evidence
- ✅ **Interactive PDF Viewer** - Click-to-navigate with highlighting
- ✅ **Agent Orchestration** - Dynamic agent chaining

### **Advanced Features**
- ✅ **Tool Calling** - Agents call Python functions
- ✅ **Session Management** - Automatic conversation history
- ✅ **Map-Reduce Summarization** - Handles 100+ page documents
- ✅ **Evidence Highlighting** - Yellow highlights on cited passages
- ✅ **Execution Tracing** - Transparent agent workflow
- ✅ **Built-in Tracing** - Debug and optimize workflows

---

## 🚀 Quick Start

### **1. Prerequisites**

- Python 3.9+
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### **2. Installation**

```bash
# Clone repository
git clone <repository-url>
cd pdf_agent_system

# Install dependencies
pip install -r requirements.txt
```

### **3. Configuration**

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=sk-your-api-key-here
```

### **4. Run Application**

```bash
streamlit run app.py
```

Access the application at: **http://localhost:8501**

---

## 📁 Project Structure

```
pdf_agent_system/
├── agents/
│   ├── __init__.py
│   ├── rag_agent.py                # RAG with retrieval tool (150 lines)
│   ├── summarization_agent.py      # Map-reduce summarization (120 lines)
│   ├── specialized_agents.py       # 3 reasoning agents (180 lines)
│   └── planner_agent.py            # Orchestrator (200 lines)
├── utils/
│   ├── __init__.py
│   ├── pdf_processor.py            # PDF extraction + chunking (180 lines)
│   └── vector_store.py             # FAISS vector database (150 lines)
├── config/
│   ├── __init__.py
│   └── settings.py                 # Configuration (50 lines)
├── app.py                          # Streamlit UI (454 lines)
├── requirements.txt                # Dependencies
├── .env.example                    # Configuration template
├── .gitignore                      # Git ignore rules
└── README.md                       # This file
```

**Total:** ~1,500 lines of production code

---

## 🎓 How It Works

### **1. OpenAI Agents SDK Integration**

Create agents using the official SDK:

```python
from agents import Agent, Runner

# Create agent
agent = Agent(
    name="RAG Agent",
    instructions="You are a RAG agent...",
    tools=[retrieve_documents]
)

# Run agent
result = Runner.run_sync(agent, "What are the findings?")
print(result.final_output)
```

### **2. Tool Functions**

Agents can call Python functions:

```python
def retrieve_documents(query: str, top_k: int = 5):
    """Retrieve relevant document chunks"""
    results = vector_store.search(query, top_k)
    return format_results(results)

# Agent automatically calls this function when needed
```

### **3. Agent Orchestration**

```python
# Planner detects intent
intent = planner.detect_intent("Compare X and Y")

# Routes to agent chain
evidence = rag_agent.retrieve(query)
comparison = comparator_agent.analyze(evidence)
```

---

## 💡 Usage Examples

### **Example 1: Question Answering**

```
User: "What are the main findings in the research paper?"

System Flow:
1. Planner detects QUERY intent
2. RAG Agent retrieves 5 relevant chunks
3. Agent generates answer with citations

Output:
Answer: "The research identifies three main findings: [1] X, [2] Y, [3] Z"
Evidence: 5 cited sources with page numbers
```

### **Example 2: Document Summarization**

```
User: "Summarize all uploaded documents"

System Flow:
1. Planner detects SUMMARIZE intent
2. Summarization Agent uses map-reduce strategy
3. Combines summaries into coherent narrative

Output:
Comprehensive summary covering all documents
```

### **Example 3: Comparative Analysis**

```
User: "Compare the methodologies across these papers"

System Flow:
1. Planner detects COMPARE intent
2. RAG Agent retrieves methodology sections
3. Comparator Agent analyzes differences

Output:
Structured comparison with specific examples
```

### **Example 4: Timeline Construction**

```
User: "Create a timeline of events mentioned"

System Flow:
1. Planner detects TIMELINE intent
2. RAG Agent retrieves temporal information
3. Timeline Builder organizes chronologically

Output:
Ordered timeline with dates and causal relationships
```

---

## ⚙️ Configuration

### **Environment Variables**

```bash
# Required
OPENAI_API_KEY=sk-your-key-here

# Optional (defaults shown)
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
SIMILARITY_THRESHOLD=0.1
```

---

## 🎯 Key Implementation Details

### **RAG Pipeline**

```
PDF Upload → Text Extraction → Chunking (1000 chars, 200 overlap)
              ↓
    Embedding (sentence-transformers)
              ↓
    FAISS Index (IndexFlatIP, cosine similarity)
              ↓
    Query → Top-K Retrieval → Context Assembly → LLM Generation
```

### **Agent Tool Calling**

```
User Query → Agent decides to call tool
             ↓
       Tool executed in Python
             ↓
       Results returned to Agent
             ↓
       Agent generates final response
```

### **Map-Reduce Summarization**

```
Long Document (100+ chunks)
    ↓
Split into batches (10 chunks each)
    ↓
Summarize each batch (Map step)
    ↓
Combine summaries (Reduce step)
    ↓
Final coherent summary
```

---

## 💰 Cost Considerations

### **OpenAI Pricing**

| Model | Input (per 1M tokens) | Output (per 1M tokens) |
|-------|----------------------|------------------------|
| gpt-4o-mini | $0.150 | $0.600 |
| gpt-4o | $2.50 | $10.00 |

### **Typical Usage**

- **Per Query:** ~2,000 input + 500 output tokens = ~$0.0006
- **Per Session:** ~10 queries = ~$0.006
- **Testing/Development:** ~$1-2 total

---

## 📊 System Metrics

| Metric | Value |
|--------|-------|
| **Total Agents** | 6 |
| **Total Code** | ~1,500 lines |
| **Vector Dimensions** | 384 |
| **Default Top-K** | 5 |
| **Chunk Size** | 1000 chars |
| **Chunk Overlap** | 200 chars |

---

## 🎉 Acknowledgments

- **OpenAI** - Agents SDK framework
- **Facebook Research** - FAISS vector search
- **Sentence Transformers** - Embedding models
- **Streamlit** - Interactive UI framework

---

<div align="center">

### ✨ Built with OpenAI Agents SDK v0.6.1 | Multi-Agent Architecture ✨

</div>

---