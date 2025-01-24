import streamlit as st
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
import sentence_transformers
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.document_loaders import WebBaseLoader
import requests
from bs4 import BeautifulSoup

#Setup Streamlit page
st.title("Insurance Policy Intelligence and Comparison Chatbot")

#LLM
api_key = "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"
if "llm" not in st.session_state:
  st.session_state.llm = ChatGroq(groq_api_key = api_key, model_name = 'llama-3.1-70b-versatile', temperature = 0.2, top_p = 0.2)

#Embedding
if "hf_embedding" not in st.session_state:
  st.session_state.hf_embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#Placeholder for user input: Web URLs
st.write("Enter the sitemap URLs:")
sitemap_urls_input = st.text_area("Sitemap URLs", height = 100, placeholder = "Enter URLs like https://example.com/sitemap.xml")

#Placeholder for user input: keywords to filter
st.write("Enter keywords to filter URLs:")
filter_urls_input = st.text_input("Keywords", placeholder = "Enter keywords like 'saral,pension'")

#Parse user input
sitemap_urls = [url.strip() for url in sitemap_urls_input.split("\n") if url.strip()]
filter_urls = [keyword.strip() for keyword in filter_urls_input.split(",") if keyword.strip()]

#URL scraping logic
def load_documents():
  if not sitemap_urls or not filter_urls:
    st.error("Error")
    return
    
  filtered_urls = []
  for sitemap_url in sitemap_urls:
    try:
      response = requests.get(sitemap_url)
      sitemap_content = response.content

      #Parse sitemap URL
      soup = BeautifulSoup(sitemap_content, 'lxml')
      urls = [loc.text for loc in soup.find_all('loc')]

      #Filter URLs
      selected_urls = [url for url in urls if any(filter in url for filter in filter_urls)]
      #Append URLs to the main list
      filtered_urls.extend(selected_urls)
    except Exception as e:
      st.error("Error loading sitemap")
      return
  docs = []
  for url in filtered_urls:
      try:
        loader = WebBaseLoader(url)
        docs.extend(loader.load())
        #st.success(f"Successfully loaded content from: {url}")
        
      except Exception as e:
        st.error("Failed to load content")
  st.session_state.docs = docs
  st.session_state.docs_loaded = True
  st.success(f"Successfully loaded content from: {len(filtered_urls)} URL(s)")
#Load the URLs when user clicks a button
if "docs_loaded" not in st.session_state:
  st.session_state.docs_loaded = False
  
if st.button("Load Documents") and not st.session_state.docs_loaded:
  load_documents()
if st.session_state.docs_loaded:
  #Text Splitting
  if "document_chunks" not in st.session_state:
    text_splitter = RecursiveCharacterTextSplitter(
          chunk_size = 1500,
          chunk_overlap  = 100,
          length_function = len,
      )
    st.session_state.document_chunks = text_splitter.split_documents(st.session_state.docs)

      #Vector db creation
  if "vector_db" not in st.session_state:
    try:
       st.session_state.vector_db = FAISS.from_documents(st.session_state.document_chunks, st.session_state.hf_embedding)
       st.write("Vector store created successfully")
    except Excpetion as e:
        st.error("Error creating vector store")


#Craft ChatPrompt Template
  prompt = ChatPromptTemplate.from_template(
      """
      You are an HDFC Life Insurance specialist who needs to answer queries based on the information provided in the websites. Please follow all the websites, and answer as per the same.
      Do not answer anything out of the website information.
      Do not skip any information as per the query asked from the context. Answer appropriately as per the query asked.
      All pointers for every questions asked should be mentioned as per the information provided on website.
      Now, being an excellent HDFC Life Insurance agent, you need to compare your policies against the other company's policies in the websites.
      Generate tabular data wherever required to classify the difference between different parameters of policies.
      I will tip you with a $1000 if the answer provided is helpful.
      <context>
      {context}
      </context>
      Question: {input}""")
  
    #Retriever from Vector store
  retriever = st.session_state.vector_db.as_retriever()

    #Stuff Document Chain Creation
  document_chain = create_stuff_documents_chain(st.session_state.llm, prompt)

    #Create a retrieval chain
  retrieval_chain = create_retrieval_chain(retriever,document_chain)

    #Input for user queries
  user_query = st.text_input("Ask a question")

    #Process the query
  if user_query:
      st.write(f"Processing query: {user_query}")
      try:
        response = retrieval_chain.invoke({"input": user_query})
        
        #Check if response is valid
        if response and 'answer' in response:
          st.write(response['answer'])
        else:
          st.error("No answer returned")
      except Exception as e:
        st.error("Error during retrieval chain execution")
