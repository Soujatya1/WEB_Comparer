import streamlit as st
from utils import load_and_split_documents, create_embeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import ChatGroq

# Streamlit UI
st.title("Website Intelligence Comparer")
st.sidebar.header("Settings")

# Input Fields
sitemap_urls = st.sidebar.text_area("Enter Sitemap URLs (comma-separated)", "").split(",")
filters = st.sidebar.text_area("Enter URL Filters (comma-separated)", "").split(",")

if st.sidebar.button("Compare Websites"):
    if not sitemap_urls:
        st.error("Please provide sitemap URLs.")
    else:
        # Load and process documents
        with st.spinner("Loading and processing documents..."):
            documents = load_and_split_documents(sitemap_urls, filters)
            st.write(f"Loaded {len(documents)} documents.")

        # Generate embeddings
        with st.spinner("Generating embeddings..."):
            embedding_model = HuggingFaceEmbeddings()
            vectorstore = FAISS.from_documents(documents, embedding_model)
            st.success("Embeddings created and indexed.")

        # Query and compare
        query = st.text_input("Enter your query", "")
        if query:
            llm = ChatGroq(groq_api_key=api_key, model_name='llama-3.1-70b-versatile', temperature=0.2, top_p=0.2)
            results = vectorstore.similarity_search_with_score(query, k=5)
            for doc, score in results:
                st.write(f"**Source:** {doc.metadata['source']}\n**Score:** {score}\n{doc.page_content[:200]}...")
