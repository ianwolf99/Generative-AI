import os
import logging
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma

PDF_DIRECTORY = os.path.join(os.path.dirname(__file__), 'pdf')
PERSIST_DIRECTORY = "persist"
PERSIST = True

# Configure loggings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

logging.info("key loaded")
try:
    
    
    if PERSIST and os.path.exists(PERSIST_DIRECTORY):
        logging.info("Persisted vectorstore found, loading from directory.")
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=HuggingFaceEmbeddings())
    else:
        logging.info("Persisted vectorstore not found, creating a new one.")
        if not os.path.exists(PDF_DIRECTORY):
            os.makedirs(PDF_DIRECTORY)
        
        loader = DirectoryLoader(PDF_DIRECTORY)
        data = loader.load()
        if not data:
            raise ValueError("No data loaded from directory.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(data)
        logging.info("Documents split into chunks.")
        
        embeddings = HuggingFaceEmbeddings()
        vectorstore = Chroma.from_documents(docs, embeddings, persist_directory=PERSIST_DIRECTORY)
        logging.info("Chroma vector index created and persisted.")
    
    
    logging.info("Trained now query")
    
except Exception as e:
    logging.error(f"An error occurred: {e}")
