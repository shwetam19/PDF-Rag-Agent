from agents import function_tool
from utils.state import global_state

@function_tool
def retrieve_documents(query: str) -> str:
    """
    Retrieve relevant document chunks based on a natural language query.
    Returns the evidence text with metadata (Document Name, Page Number).
    """
    print(f"🛠️ Tool Call: Retrieving for '{query}'...")
    
    if not global_state.vector_store:
        return "Error: No documents have been indexed yet."

    # Using the vector_store from global state
    retrieved = global_state.vector_store.search(query=query, top_k=5)
    
    if not retrieved:
        return "No relevant information found in the documents."
    
    # Format context for the LLM
    context_parts = []
    for i, item in enumerate(retrieved, 1):
        context_parts.append(
            f"[Source {i}] Doc: {item['doc_name']}, Page: {item['page_num']}\n"
            f"Content: {item['chunk'].text}\n"
        )
    
    return "\n---\n".join(context_parts)

@function_tool
def get_full_document_content() -> str:
    """
    Retrieves the text content of all uploaded documents for summarization.
    """
    print("🛠️ Tool Call: Reading full documents...")
    
    if not global_state.vector_store or not global_state.vector_store.chunks:
        return "No documents available."

    # Combine text (Simplified for demonstration)
    all_text = [chunk.text for chunk in global_state.vector_store.chunks]
    
    # Cap content to avoid context limits if necessary
    full_text = "\n\n".join(all_text[:300]) 
    return full_text