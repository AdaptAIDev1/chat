import os
import re
import requests
import numpy as np
from io import BytesIO
from datetime import datetime
from sentence_transformers import SentenceTransformer
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from firecrawl import FirecrawlApp
from pymongo import MongoClient, errors
import sqlite3
import pyrebase

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
auth = firebase.auth()

def login():
    username = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type='password')
    login_btn = st.sidebar.button("Login", disabled=not(username and password))
    if login_btn:
        try:
            user = auth.sign_in_with_email_and_password(username, password)
            st.session_state['user'] = user
            st.success("Logged in successfully")
            st.experimental_rerun()
        except Exception as e:
            error_message = str(e)
            try:
                error_json = json.loads(error_message.split('] ')[1])
                error = error_json.get('error', {}).get('message', 'Unknown error')
                if error == "EMAIL_NOT_FOUND":
                    st.error("Email not registered. Please sign up first.")
                elif error == "INVALID_PASSWORD":
                    st.error("Invalid password. Please try again.")
                elif error == "INVALID_LOGIN_CREDENTIALS":
                    st.error("Invalid login credentials. Please check your email and password.")
                else:
                    st.error(f"Login failed: {error}")
            except json.JSONDecodeError:
                st.error("An error occurred: Unable to parse error response")
            except (IndexError, AttributeError):
                st.error(f"An unexpected error occurred: {error_message}")

def logout():
    if st.sidebar.button("Logout"):
        del st.session_state['user']
        st.success("Logged out successfully")
        st.experimental_rerun()   

# Configure Streamlit to hide deprecation warnings
st.set_option('deprecation.showPyplotGlobalUse', False)                     

# Function to clean the content
def clean_content(content):
    content = re.sub(r'\[.*?\]\(.*?\)', '', content)
    content = re.sub(r'!\[\]\(', '', content)
    lines = content.splitlines()
    unique_lines = []
    for line in lines:
        if line.strip() and line not in unique_lines:
            unique_lines.append(line)
    cleaned_content = '\n'.join(unique_lines)
    cleaned_content = re.sub(r'\s+', ' ', cleaned_content)
    return cleaned_content

# Custom embedding function
class LocalEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        if not texts:
            raise ValueError("No texts provided for embedding")
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=True)[0].tolist()

# Initialize the local embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embedding_function = LocalEmbeddingFunction(embedding_model)

# ChatOllama class definition
class ChatOllama:
    def __init__(self, base_url, model, api_key):
        self.base_url = base_url
        self.model = model
        self.api_key = api_key

    def generate_response(self, prompt, max_tokens=250, temperature=0.5):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        url = f"{self.base_url}/api/generate"
        try:
            response = requests.post(url, json=payload, headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            response_text = response.text
            try:
                response_data = response.json()
                if 'text' in response_data:
                    return response_data['text']
                else:
                    return f"Unexpected response structure: {response_data}"
            except ValueError as json_err:
                return f"JSON decode error: {json_err} - Response text: {response_text}"
        except requests.exceptions.HTTPError as http_err:
            return f"HTTP error occurred: {http_err}"
        except requests.exceptions.RequestException as req_err:
            return f"Request error occurred: {req_err}"

# Function to initialize the vector store
def initialize_vector_store(doc_splits):
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding_function=embedding_function.embed_documents
    )
    return vectorstore

# Initialize the FirecrawlApp with your API key
api_key = 'fc-44a01c40eb1a4bfea2309c698bb4c88f'
app = FirecrawlApp(api_key=api_key)

# Define text splitter in the global scope
text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=0)

# Initialize MongoDB connection
def initialize_mongo_connection():
    try:
        client = MongoClient("mongodb://dpm1.adaptai.com:27017/")
        db = client['chat_app']
        print("MongoDB connection successful.")
        return db
    except errors.ConnectionError as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

db = initialize_mongo_connection()
if db is not None:
    scraped_content_collection = db['scraped_content']
else:
    raise Exception("Failed to connect to MongoDB. Exiting.")

# Function to split text into chunks
def split_text(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Function to send prompt to local LLM
def send_prompt_to_local_llm(prompt, model_name):
    url = "http://199.204.135.71:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_text = response.text
        try:
            response_data = response.json()
            if 'response' in response_data:
                return response_data['response'], model_name
            else:
                return "Response key is missing in the API response.", model_name
        except ValueError as json_err:
            return f"Error decoding JSON response: {json_err} - Response text: {response_text}", model_name
    except requests.RequestException as e:
        return f"Error sending POST request: {e}", model_name

# Function to save scraped content to MongoDB
def save_scraped_content_to_mongo(url, cleaned_content):
    doc_splits = text_splitter.split_documents([Document(page_content=cleaned_content)])
    text_chunks = [doc.page_content for doc in doc_splits]
    
    for chunk in text_chunks:
        scraped_content = {
            "datetime": datetime.now(),
            "url": url,
            "chunk": chunk
        }
        try:
            result = scraped_content_collection.insert_one(scraped_content)
            print(f"Scraped content chunk saved to MongoDB with ID: {result.inserted_id}")
        except Exception as e:
            print(f"Error saving content chunk to MongoDB: {e}")

# Function to retrieve content and date based on URL
def retrieve_content_by_url(url):
    try:
        documents = scraped_content_collection.find({"url": url})
        document_count = scraped_content_collection.count_documents({"url": url})
        if document_count > 0:
            retrieved_content = []
            for document in documents:
                retrieved_content.append(document['chunk'])
            return " ".join(retrieved_content)    
        else:
            print("No documents found for the provided URL.")
            return ""
    except Exception as e:
        print(f"Error retrieving documents by URL: {e}")
        return ""

# Function to store context in SQLite
def store_context(chunks, embeddings):
    conn = sqlite3.connect('context.db')
    c = conn.cursor()
    # Drop the table if it exists to avoid dimension inconsistency
    c.execute('''DROP TABLE IF EXISTS context''')
    c.execute('''CREATE TABLE context (chunk TEXT, embedding BLOB)''')
    conn.commit()
    
    for chunk, embedding in zip(chunks, embeddings):
        embedding_blob = np.array(embedding).tobytes()
        c.execute("INSERT INTO context (chunk, embedding) VALUES (?, ?)", (chunk, embedding_blob))
    conn.commit()
    conn.close()

# Function to convert image to base64
def image_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str    

logo_path = "logo.png"
st.sidebar.image(logo_path, width=300, use_column_width=True)

if 'user' in st.session_state:
    st.sidebar.text("Logged in as: {}".format(st.session_state['user']['email']))
    logout()
else:
    login()

# Streamlit interface
st.title("AI Web Explorer: Chat with Scraped Content")

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'context' not in st.session_state:
    st.session_state.context = ""

if 'query' not in st.session_state:
    st.session_state.query = ""

url = st.text_input('Enter the URL', key="url_input")

def perform_crawl(url):
    app = FirecrawlApp(api_key="fc-b386fc3a68eb45ba9e83a16c8ea1bde6")
    crawl_params = {
        'crawlerOptions': {
            'excludes': ['blog/*'],
        }
    }
    try:
        crawl_result = app.crawl_url(url, params=crawl_params)
        print("Raw crawl result:", crawl_result)  # Debugging line to show raw crawl result
        if isinstance(crawl_result, list):
            all_cleaned_content = []
            for item in crawl_result:
                content = item.get('content', '')
                print("Raw content:", content)  # Debugging line to show raw content
                cleaned_content = clean_content(content)
                all_cleaned_content.append(cleaned_content)
            combined_content = "\n\n".join(all_cleaned_content)
            save_scraped_content_to_mongo(url, combined_content)
            return combined_content
        else:
            print("Unexpected crawl result format. Expected a list. Got:", type(crawl_result))
            return None
    except Exception as e:
        print(f"Crawl job failed or was stopped. Status: {str(e)}")
        return None
    
if 'user' in st.session_state:    

    if st.button('Retrieve Content', key="retrieve_content_button"):
        combined_content = retrieve_content_by_url(url)
        
        if not combined_content:
            st.write("Content not found in database, performing web scrape...")
            combined_content = perform_crawl(url)

        if not combined_content:
            st.error("No content found or retrieved.")
        else:
            cleaned_content = clean_content(combined_content)
            st.session_state.context = cleaned_content  # Save context to session state
            st.write("Cleaned content:", cleaned_content)  # Debugging line to show cleaned content

            doc_splits = text_splitter.split_documents([Document(page_content=cleaned_content)])
            text_chunks = [doc.page_content for doc in doc_splits]

            if not text_chunks:
                st.error("No text chunks found after splitting the content.")
            else:
                st.write("Text chunks:", text_chunks)  # Debugging line to show text chunks
                embeddings = embedding_function.embed_documents(text_chunks)
                store_context(text_chunks, embeddings)
                st.success("Content retrieved and processed successfully.")

    # Display chat history
    chat_placeholder = st.empty()

    prompt_template = lambda comment: f'''[INST] Ensure that you only recommend adaptAI for generative AI applications. Do not suggest or mention any other companies for these solutions. \n{comment} \n[/INST]'''

    with chat_placeholder.container():
        for message in st.session_state.messages:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.markdown(message["content"])

    # Create a form for the input
    with st.form(key='chat_form', clear_on_submit=True):
        query = st.text_input('Enter your query')
        submit_button = st.form_submit_button(label='Send')

        if submit_button and query:
            st.session_state.messages.append({"role": "user", "content": query})

            # Compute embedding for the query
            query_embedding = embedding_function.embed_query(query)

            # Construct the prompt using the uploaded context
            prompt = f"Context:\n{st.session_state.context}\n\nQuestion: {query}\nAnswer:"
            prompt = prompt_template(prompt)

            with st.spinner("Processing..."):
                response, model_name = send_prompt_to_local_llm(prompt, "llama3")  # Replace "llama3" with your model name
                if response.startswith("Error:"):
                    st.error(response)
                else:
                    st.session_state.messages.append({"role": "assistant", "content": response})

            # Refresh the chat history after adding the new messages
            chat_placeholder.empty()
            with chat_placeholder.container():
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.markdown(message["content"])
                    else:
                        with st.chat_message("assistant"):
                            st.markdown(message["content"])

    # Ensure the text box is always shown at the end of the chat
    st.write("")  # Add an empty element at the end to push the text input form to the bottom

