import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModel
import faiss
import torch

# Function to load documents from URL
def load_documents_from_url(url):
    """Loads documents from a given URL."""
    try:
        loader = WebBaseLoader(web_path=url)
        return loader.load()
    except Exception as e:
        return []

# Function to process and split documents
def load_and_split_documents(urls, filters):
    """Loads and processes documents from sitemaps."""
    loaded_docs = []
    for sitemap_url in urls:
        try:
            response = requests.get(sitemap_url)
            sitemap_content = response.content
            soup = BeautifulSoup(sitemap_content, 'xml')
            urls = [loc.text for loc in soup.find_all('loc')]

            selected_urls = [url for url in urls if any(filter in url for filter in filters)]
            for url in selected_urls:
                docs = load_documents_from_url(url)
                for doc in docs:
                    doc.metadata["source"] = url
                loaded_docs.extend(docs)
        except Exception as e:
            print(f"Error processing sitemap {sitemap_url}: {e}")
    return loaded_docs

# Function to create embeddings
def create_embeddings(loaded_docs, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    """Creates embeddings using HuggingFace model."""
    tokenizer = AutoTokenizer.from_pretrained(embedding_model)
    model = AutoModel.from_pretrained(embedding_model)
    
    def embed_text(text):
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            embeddings = model(**inputs).last_hidden_state.mean(dim=1)
        return embeddings.numpy()
    
    embeddings = [embed_text(doc.page_content) for doc in documents]
    return embeddings
