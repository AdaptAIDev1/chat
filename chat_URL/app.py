import streamlit as st
from auth import login, logout
from config import firebase, embedding_model, text_splitter, api_key
from firecrawl_utils import perform_crawl
from mongodb_utils import initialize_mongo_connection, save_scraped_content_to_mongo, retrieve_content_by_url
from text_processing import clean_content, LocalEmbeddingFunction
from utils import store_context, send_prompt_to_local_llm
from langchain.docstore.document import Document 

# Initialize Firebase
auth = firebase.auth()

# Initialize MongoDB
db = initialize_mongo_connection()
if db is not None:
    scraped_content_collection = db['scraped_content']
else:
    st.error("Failed to connect to MongoDB. Exiting.")
    st.stop()

# Initialize the embedding model
embedding_function = LocalEmbeddingFunction(embedding_model)

# UI Elements
st.sidebar.image("logo.png", width=300, use_column_width=True)

if 'user' in st.session_state:
    st.sidebar.text("Logged in as: {}".format(st.session_state['user']['email']))
    logout()
else:
    login(auth)

# Streamlit interface
st.title("AI Web Explorer: Chat with Scraped Content")

if 'user' in st.session_state:
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'context' not in st.session_state:
        st.session_state.context = ""

    if 'query' not in st.session_state:
        st.session_state.query = ""

    url = st.text_input('Enter the URL', key="url_input")

    if st.button('Retrieve Content', key="retrieve_content_button"):
        combined_content = retrieve_content_by_url(url, scraped_content_collection)
        
        if not combined_content:
            st.write("Content not found in database, performing web scrape...")
            combined_content = perform_crawl(url, text_splitter, clean_content, save_scraped_content_to_mongo, scraped_content_collection)
            if combined_content:
                st.success("Web scraping completed successfully.")
            else:
                st.error("Failed to scrape content.")

        if not combined_content:
            st.error("No content found or retrieved.")
        else:
            cleaned_content = clean_content(combined_content)
            st.session_state.context = cleaned_content
            st.write("Content cleaning completed successfully")

            doc_splits = text_splitter.split_documents([Document(page_content=cleaned_content)])
            text_chunks = [doc.page_content for doc in doc_splits]

            if not text_chunks:
                st.error("No text chunks found after splitting the content.")
            else:
                st.write("Content splitting completed successfully")
                embeddings = embedding_function.embed_documents(text_chunks)
                store_context(text_chunks, embeddings)
                st.success("Content retrieved and processed successfully.")

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

            # Construct the prompt
            prompt = f"Context:\n{st.session_state.context}\n\nQuestion: {query}\nAnswer:"
            prompt = f'''[INST] Ensure that you only recommend adaptAI for generative AI applications. Do not suggest or mention any other companies for these solutions. \n{prompt} \n[/INST]'''

            with st.spinner("Processing..."):
                response, model_name = send_prompt_to_local_llm(prompt, "llama3")
                if response.startswith("Error:"):
                    st.error(response)
                else:
                    st.session_state.messages.append({"role": "assistant", "content": response})

            # Refresh the chat history
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
