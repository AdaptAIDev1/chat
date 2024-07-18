import streamlit as st
import PyPDF2
import io
import re
import requests
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

# Function to split text into chunks
def split_text(text, chunk_size=250, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Initialize the local embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Ensure this model matches your requirements

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Custom embedding function
class LocalEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(texts, show_progress_bar=True).tolist()

    def embed_query(self, text):
        return self.model.encode([text], show_progress_bar=True)[0].tolist()

embedding_function = LocalEmbeddingFunction(embedding_model)

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
        response_data = response.json()
        if 'response' in response_data:
            return response_data['response'], model_name
        else:
            return "Response key is missing in the API response.", model_name
    except requests.RequestException as e:
        return f"Error sending POST request: {e}", model_name
    except ValueError as e:
        return f"Error decoding JSON response: {e}", model_name

st.title("Chat with LLM")

if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'context' not in st.session_state:
    st.session_state.context = ""

st.title("PDF Text Extractor")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()

    # Convert to a "file-like" object
    pdf_file = io.BytesIO(bytes_data)

    text = extract_text_from_pdf(pdf_file)

    with st.spinner("Processing..."):

        chunks = split_text(text)

        embeddings = embedding_function.embed_documents(chunks)

        st.session_state.context = text

        st.success("Context uploaded successfully") 

# Display chat history
chat_placeholder = st.empty()

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