class GlobalState:
    """
    Singleton to hold shared resources (VectorStore) so standalone 
    Agent Tools can access data loaded by the Streamlit App.
    """
    vector_store = None

# Global instance
global_state = GlobalState()