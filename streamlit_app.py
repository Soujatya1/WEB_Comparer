import streamlit as st
from langchain.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_groq import ChatGroq

# Hardcoded Sitemaps and Filter Words
SITEMAP_URLS = [
    "https://www.hdfclife.com/universal-sitemap.xml",
    "https://www.reliancenipponlife.com/sitemap.xml",
]
FILTER_WORDS = ["retirement"]

# Backend Function to Load and Process Sitemaps
@st.cache_data
def process_sitemaps(sitemap_urls, filter_words):
    filtered_urls = []
    loaded_docs = []

    for sitemap_url in sitemap_urls:
        try:
            response = requests.get(sitemap_url)
            sitemap_content = response.content

            # Parse sitemap URL
            soup = BeautifulSoup(sitemap_content, 'xml')
            urls = [loc.text for loc in soup.find_all('loc')]

            # Filter URLs
            selected_urls = [url for url in urls if any(filter_word in url for filter_word in filter_words)]
            filtered_urls.extend(selected_urls)

            for url in selected_urls:
                try:
                    loader = WebBaseLoader(url)
                    docs = loader.load()
                    for doc in docs:
                        doc.metadata["source"] = url
                    loaded_docs.extend(docs)
                except Exception as e:
                    st.warning(f"Error loading {url}: {e}")

        except Exception as e:
            st.warning(f"Error processing sitemap {sitemap_url}: {e}")

    return loaded_docs, filtered_urls


# Streamlit UI
st.title("Website Intelligence")

# API Key Input
api_key = st.text_input("Enter API Key:", type="password")

if st.button("Load and Process"):
    # Call Backend Function with Hardcoded Sitemaps and Filter Words
    loaded_docs, filtered_urls = process_sitemaps(SITEMAP_URLS, FILTER_WORDS)

    st.write(f"Filtered URLs: {len(filtered_urls)}")
    st.write(f"Loaded documents: {len(loaded_docs)}")

    if api_key:
        # Initialize LLM and Embeddings
        llm = ChatGroq(groq_api_key=api_key, model_name="llama-3.1-70b-versatile", temperature=0.2, top_p=0.2)
        hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        # Text Splitting
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        document_chunks = text_splitter.split_documents(loaded_docs)
        st.write(f"Number of chunks: {len(document_chunks)}")

        # Vector Database
        vector_db = FAISS.from_documents(document_chunks, hf_embedding)

        # Prompt Template
        prompt = ChatPromptTemplate.from_template(
            """
            You are a Life Insurance specialist who needs to answer queries based on the information provided in the websites only. Please follow all the websites, and answer as per the same.

            Do not answer anything except from the website information which has been entered. Please do not skip any information from the tabular data in the website.

            Do not skip any information from the context. Answer appropriately as per the query asked.

            Now, being an excellent Life Insurance agent, you need to compare your policies against the other company's policies in the websites, if asked.

            Generate tabular data wherever required to classify the difference between different parameters of policies.

            I will tip you with a $1000 if the answer provided is helpful.
            
            <context>
            {context}
            </context>
            
            Question: {input}
            """
        )

        # Document Chain and Retriever
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = vector_db.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Query Interface
        query = st.text_input("Enter your query:")
        if st.button("Get Answer"):
            if query:
                response = retrieval_chain.invoke({"input": query})
                st.write("Response:")
                st.write(response['answer'])
