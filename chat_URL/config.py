import os
from sentence_transformers import SentenceTransformer
import pyrebase
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Set the environment variable to avoid OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure Firebase
firebase_config = {
    "apiKey": "AIzaSyAL9L7uLqlO2Z2RVny6uFAr4j72ix2LoI8",
    "authDomain": "streamlit-test-5ff43.firebaseapp.com",
    "databaseURL": "https://streamlit-test-5ff43.firebaseio.com",
    "projectId": "streamlit-test-5ff43",
    "storageBucket": "streamlit-test-5ff43.appspot.com",
    "messagingSenderId": "356923002998",
    "appId": "1:356923002998:web:71792a47dc65acfd4a4f57",
    "measurementId": "G-M964V5LBPQ"
}

firebase = pyrebase.initialize_app(firebase_config)

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)

# Firecrawl API key
api_key = 'fc-44a01c40eb1a4bfea2309c698bb4c88f'
