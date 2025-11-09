# app.py (Streamlit version, silent FAISS loading)
import streamlit as st
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
import os

# ----------------------------
# Initialize FAISS / RAG
# ----------------------------
@st.cache_resource
def init_vectorstore_and_rag(data_dir="data", persist_dir="faiss_store"):
    """
    Initialize FAISS vector store and RAG system.
    Cached so it only runs once.
    """
    # Load documents
    docs = load_all_documents(data_dir)
    
    # Initialize FAISS store
    store = FaissVectorStore(persist_dir)
    
    # Check if FAISS index exists
    faiss_index_path = os.path.join(persist_dir, "faiss.index")
    
    if os.path.exists(faiss_index_path):
        # Silently load existing index (no UI message)
        store.load()
    else:
        # Silently build and save index
        store.build_from_documents(docs)
        store.save()
    
    # Initialize RAG
    rag_search = RAGSearch(persist_dir=persist_dir)
    return rag_search

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="📚 Document Q&A", layout="wide")
st.title("📚 RAG Document Q&A")
st.write(
    "Ask questions about your documents. "
    "The system retrieves relevant content and summarizes it using an LLM."
)

# Initialize RAG system (FAISS + LLM)
rag_search = init_vectorstore_and_rag()

# Query input
query = st.text_input("Enter your question here:")

# Search button
if st.button("Search"):
    if query.strip():
        with st.spinner("Searching documents and summarizing..."):
            try:
                top_k = 5  # fixed number of retrieved chunks
                # Perform RAG search + summary
                summary = rag_search.search_and_summarize(query, top_k=top_k)
                st.subheader("Summary:")
                st.write(summary)
            except Exception as e:
                st.error(f"Error during RAG search: {e}")
    else:
        st.warning("Please enter a question.")
