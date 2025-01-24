import streamlit as st
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup

# Streamlit UI
st.title("Website Intelligence")

# Initialize session state variables
if "loaded_docs" not in st.session_state:
    st.session_state.loaded_docs = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

sitemap_urls_input = st.text_area("Enter sitemap URLs (one per line):")
filter_words_input = st.text_area("Enter filter words (one per line):")

if st.button("Load and Process"):
    sitemap_urls = sitemap_urls_input.splitlines()
    filter_urls = filter_words_input.splitlines()

    all_urls = []
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
            selected_urls = [url for url in urls if any(filter in url for filter in filter_urls)]

            # Append URLs to the main list
            filtered_urls.extend(selected_urls)

            for url in filtered_urls:
                try:
                    st.write(f"Loading URL: {url}")
                    loader = WebBaseLoader(url)
                    docs = loader.load()

                    for doc in docs:
                        doc.metadata["source"] = url

                    loaded_docs.extend(docs)
                    st.write("Successfully loaded document")
                except Exception as e:
                    st.write(f"Error loading {url}: {e}")

        except Exception as e:
            st.write(f"Error processing sitemap {sitemap_url}: {e}")

    st.write(f"Loaded documents: {len(loaded_docs)}")

    # Store loaded documents in session state
    st.session_state.loaded_docs = loaded_docs

llm = ChatGroq(groq_api_key="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", model_name='llama-3.1-70b-versatile', temperature=0.2, top_p=0.2)

# Embedding
hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-MiniLM-L3-v2")

# Craft ChatPrompt Template
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

            Question: {input}"""
        )

# Text Splitting
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)

document_chunks = text_splitter.split_documents(loaded_docs)
st.write(f"Number of chunks: {len(document_chunks)}")

# Vector database storage
vector_db = FAISS.from_documents(document_chunks, hf_embedding)

# Stuff Document Chain Creation
document_chain = create_stuff_documents_chain(llm, prompt)

        # Retriever from Vector store
retriever = vector_db.as_retriever()

        # Create a retrieval chain
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# Save retriever and retrieval_chain to session state
st.session_state.retriever = retriever
st.session_state.retrieval_chain = retrieval_chain

query = st.text_input("Enter your query:")
if st.button("Get Answer"):
    if query:
        response = st.session_state.retrieval_chain.invoke({"input": query})
        st.write("Response:")
        st.write(response['answer'])
    else:
        st.write("No documents loaded. Please load and process the sitemap first.")
