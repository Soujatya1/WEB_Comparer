import streamlit as st
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import WebBaseLoader
import hashlib
 
# Initialize session state variables
if 'loaded_docs' not in st.session_state:
  st.session_state['loaded_docs'] = []
if 'vector_db' not in st.session_state:
  st.session_state['vector_db'] = None
if 'retrieval_chain' not in st.session_state:
  st.session_state['retrieval_chain'] = None

# Initialize embedding cache
if 'embedding_cache' not in st.session_state:
    st.session_state['embedding_cache'] = {}
 
# Streamlit UI
st.title("Website Intelligence")

api_key = "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"

#website_urls_input = st.text_area("Enter website URLs (one per line):")

sitemap_urls_input = st.text_area("Enter sitemap URLs (one per line):")
filter_words_input = st.text_area("Enter filter words (one per line):")

# Cache the loading and processing of URLs and documents
@st.cache_resource
def load_and_process_documents(sitemap_urls):
    loaded_docs = []
    for url in sitemap_urls:
        try:
            loader = WebBaseLoader(url)
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = url
            loaded_docs.extend(docs)
        except Exception as e:
            st.write(f"Error loading {url}: {e}")
    return loaded_docs

@st.cache_resource
def create_vector_db(_docs, _hf_embedding, existing_vector_db=None):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len,
    )
    
    document_chunks = text_splitter.split_documents(_docs)
    
    if existing_vector_db:
        # Add new chunks to the existing vector DB
        existing_vector_db.add_documents(document_chunks)
        return existing_vector_db, len(document_chunks)
    else:
        # Create new FAISS vector database
        vector_db = FAISS.from_documents(document_chunks, _hf_embedding)
        return vector_db, len(document_chunks)

def get_cache_key(urls):
    """
    Generates a unique cache key based on the input website URLs.
    Uses a hash to ensure a unique key for different sets of inputs.
    """
    # Combine URLs into a single string
    combined_input = ''.join(urls)
    # Generate a hash value from the combined input
    cache_key = hashlib.md5(combined_input.encode('utf-8')).hexdigest()
    return cache_key

# Load and process documents
if st.button("Load and Process"):
    sitemap_urls = sitemap_urls_input.splitlines()
    cache_key = get_cache_key(sitemap_urls)
    
    # Check if embeddings exist in cache for previous URLs
    if cache_key in st.session_state['embedding_cache']:
        # Load from cache
        cached_vector_db, cached_num_chunks, cached_urls = st.session_state['embedding_cache'][cache_key]
        st.session_state['vector_db'] = cached_vector_db
        st.write(f"Loaded cached embeddings for {len(cached_urls)} URLs. Number of chunks: {cached_num_chunks}")
    else:
        # Process and embed all URLs (no cache available)
        loaded_docs = load_and_process_documents(website_urls)
        hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state['vector_db'], num_chunks = create_vector_db(loaded_docs, hf_embedding)
        
        # Cache the embeddings and document URLs
        st.session_state['embedding_cache'][cache_key] = (st.session_state['vector_db'], num_chunks, website_urls)
        st.write(f"Processed and embedded {len(website_urls)} URLs. Number of chunks: {num_chunks}")
    
    # LLM Initialization and prompt setup
    if api_key:
        llm = ChatGroq(groq_api_key=api_key, model_name='llama-3.1-70b-versatile', temperature=0.2, top_p=0.2)
        
        # Create prompt and retrieval chain
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Website Intelligence specialist who needs to answer queries based on the information provided in the websites based on insurance policies.
            Two or three or more sitemaps will be provided for different insurance companies and you need to compare between the asked policy and suggest the better amongst them.
            <context>
            {context}
            </context>
            Question: {input}"""
        )
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state['vector_db'].as_retriever()
        st.session_state['retrieval_chain'] = create_retrieval_chain(retriever, document_chain)

# Query Section
query = st.text_input("Enter your query:")
if st.button("Get Answer") and query:
    if st.session_state['retrieval_chain']:
        response = st.session_state['retrieval_chain'].invoke({"input": query})
        st.write("Response:")
        st.write(response['answer'])
    else:
        st.write("Please load and process documents first.")
