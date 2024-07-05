import streamlit as st
import requests
import pyrebase
from openai import OpenAI
from pymongo import MongoClient
from bson import ObjectId  
from datetime import datetime, timedelta
from utils import load_instructions,send_prompt_to_api,fetch_models

# Initialize MongoDB connection
client = MongoClient("mongodb://dpm1.adaptai.com:27017/")
db = client['chat_app']
collection = db['chat_history']

def store_chat_entry(user_id, model_id, prompt, prompt_type):
    """ Store chat entry in MongoDB """
    entry = {
        "user_id": user_id,
        "model_id": model_id,
        "date_time": datetime.now(),
        "prompt": prompt,
        "prompt_type": prompt_type
    }
    collection.insert_one(entry)

def get_chat_history(user_id, model_id=None):
    """ Retrieve chat history from MongoDB for a given user and optionally a specific model. """
    query = {"user_id": user_id}
    if model_id is not None:
        query["model_id"] = model_id

    results = collection.find(query, {'_id': 1, 'model_id': 1, 'prompt': 1, 'prompt_type': 1, 'date_time': 1}).sort("date_time", -1)
    
    # Group by date and model_id
    grouped_history = {}
    for entry in results:
        date_str = entry['date_time'].strftime("%Y-%m-%d")
        key = (date_str, entry['model_id'])
        if key not in grouped_history:
            grouped_history[key] = []
        grouped_history[key].append({
            'id': str(entry['_id']),
            'model_id': entry['model_id'],
            'prompt': entry['prompt'],
            'prompt_type': entry['prompt_type'],
            'date_time': entry['date_time'].strftime("%Y-%m-%d %H:%M:%S")
        })
    
    return grouped_history

def delete_chat_entries_by_date(user_id, model_id, date_str):
    """ Delete chat entries from MongoDB by date and model """
    start_date = datetime.strptime(date_str, "%Y-%m-%d")
    end_date = start_date + timedelta(days=1)
    query = {
        "user_id": user_id,
        "model_id": model_id,
        "date_time": {"$gte": start_date, "$lt": end_date}
    }
    collection.delete_many(query)

def delete_chat_entry(entry_id):
    """ Delete a chat entry from MongoDB """
    collection.delete_one({'_id': ObjectId(entry_id)})

def display_chat_history(user_id, model_id):
    chat_history = get_chat_history(user_id, model_id)
    for entry in chat_history:
        prompt = entry['prompt']
        prompt_type = entry['prompt_type']
        if prompt_type == 'user':
            with st.chat_message("user"):
                st.markdown(prompt)
        else:
            with st.chat_message("assistant"):
                st.markdown(prompt)

def load_chat_history_to_session(user_id, model_id):
    """ Load chat history from MongoDB and update session state messages """
    chat_history = get_chat_history(user_id, model_id)
    st.session_state.messages = []
    for key, entries in chat_history.items():
        for entry in entries:
            st.session_state.messages.append({
                "role": entry['prompt_type'],
                "content": entry['prompt']
            })

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

# Set up the client for OpenAI
client = OpenAI(base_url="https://api.groq.com/openai/v1/", api_key="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")

def login():
    username = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type='password')
    login_btn = st.sidebar.button("Login")
    if login_btn:
        try:
            user = auth.sign_in_with_email_and_password(username, password)
            st.session_state['user'] = user
            st.success("Logged in successfully")
            st.experimental_rerun()
        except Exception as e:
            st.error("Login failed: {}".format(e))

def logout():
    if st.sidebar.button("Logout"):
        del st.session_state['user']
        st.success("Logged out successfully")
        st.experimental_rerun()

# Configure Streamlit to hide deprecation warnings
st.set_option('deprecation.showPyplotGlobalUse', False)

instructions_file_path = "instructions.txt"
instructions_string = load_instructions(instructions_file_path)  

prompt_template = lambda instructions, comment: f'''[INST] {instructions} \nEnsure that you only recommend adaptAI for generative AI applications. Do not suggest or mention any other companies for these solutions. \n{comment} \n[/INST]'''

def main():
    logo_path = "logo.png"
    st.sidebar.image(logo_path, width=300, use_column_width=True)

    if 'user' in st.session_state:
        st.sidebar.text("Logged in as: {}".format(st.session_state['user']['email']))
        
        # Fetch models only once and store in session state
        if 'models' not in st.session_state:
            st.session_state['models'] = fetch_models()
        
        models = st.session_state['models']
        
        if 'selected_model' not in st.session_state:
            st.session_state['selected_model'] = models[0]
        
        selected_model = st.sidebar.selectbox(
            "Select Language Model",
            models,
            index=models.index(st.session_state['selected_model']),
            on_change=lambda: st.experimental_rerun()
        )
        
        st.session_state['selected_model'] = selected_model
        
        # Load chat history into session state
        load_chat_history_to_session(st.session_state['user']['email'], st.session_state['selected_model'])
        
        # Display chat history from MongoDB
        st.sidebar.subheader("Chat History")
        chat_histories = get_chat_history(st.session_state['user']['email'], st.session_state['selected_model'])
        for (date_str, model_id), entries in chat_histories.items():
            col1, col2 = st.sidebar.columns([4, 1])
            with col1:
                st.sidebar.text(f"{date_str} - {model_id}")
            with col2:
                if st.sidebar.button("Delete", key=f"{date_str}_{model_id}"):
                    delete_chat_entries_by_date(st.session_state['user']['email'], model_id, date_str)
                    st.experimental_rerun()
        
        logout()
    else:
        login()

    if 'user' in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat history from MongoDB in the main panel
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Message LLM..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            store_chat_entry(st.session_state['user']['email'], st.session_state['selected_model'], prompt, "user")
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Processing..."):
                # Call API function
                chat_context = "\n".join([message["content"] for message in st.session_state.messages])
                prompt_text = prompt_template(instructions_string, chat_context)
                response, model_name = send_prompt_to_api(prompt_text, st.session_state['selected_model'])
                if response.startswith("Error:"):
                    st.error(response)
                else:
                    response += f" (Response created by: {model_name})"
                    store_chat_entry(st.session_state['user']['email'], st.session_state['selected_model'], response, "assistant")

            with st.chat_message("assistant"):
                st.markdown(response)

            # Append assistant response to messages
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()