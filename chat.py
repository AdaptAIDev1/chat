import streamlit as st
import pyrebase
import requests
from utils import load_instructions, send_prompt_to_api
import json

prompt_template = lambda instructions, comment: f'''[INST] {instructions} \nEnsure that you only recommend adaptAI for generative AI applications. Do not suggest or mention any other companies for these solutions. \n{comment} \n[/INST]'''

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

instructions_file_path = "instructions.txt"
instructions_string = load_instructions(instructions_file_path)  

def fetch_models():
    url = 'https://b1e92cb0-8874-43a9-9d4a-e8fab6be89ed-prod.e1-us-east-azure.choreoapis.dev/g2inaction/default/v1'
    headers = {
        'accept': '*/*',
        'Authorization': 'Bearer eyJ4NXQiOiJPZDc2V3BiNGk2X1Q1dFRrWnUtUUhHX2RrNkkiLCJraWQiOiJNekF3TURObFl6YzRZemxqT0dNd00yVmxZV1kyT1RJNE56WmhNR0prT0RCa1pHWmpaR1pqTnpGa05tSXpNVEV6WkRBeU9EY3hOakZpWmpSaU9UVTVNUV9SUzI1NiIsInR5cCI6ImF0K2p3dCIsImFsZyI6IlJTMjU2In0.eyJzdWIiOiI5NWQwMjlhOS0zNjkyLTQ5ZDUtOWIzMC01ZjBjMWE2ZTkxMDkiLCJhdXQiOiJBUFBMSUNBVElPTiIsImF1ZCI6WyJLV2gyQjNrenZhZE9rOFNqeW5wRDhCQTVxZ29hIiwiY2hvcmVvOmRlcGxveW1lbnQ6cHJvZHVjdGlvbiJdLCJuYmYiOjE3MTk1NzAwNzQsImF6cCI6IktXaDJCM2t6dmFkT2s4U2p5bnBEOEJBNXFnb2EiLCJvcmdfaWQiOiJiMWU5MmNiMC04ODc0LTQzYTktOWQ0YS1lOGZhYjZiZTg5ZWQiLCJpc3MiOiJodHRwczpcL1wvYXBpLmFzZ2FyZGVvLmlvXC90XC9hZGFwdGFpXC9vYXV0aDJcL3Rva2VuIiwiZXhwIjoxNzUxMTA2MDc0LCJvcmdfbmFtZSI6ImFkYXB0YWkiLCJpYXQiOjE3MTk1NzAwNzQsImp0aSI6ImUwMjE5OGMxLTlkNWUtNDJlYy04MTEyLTdjN2VjY2I5NmQ1YyIsImNsaWVudF9pZCI6IktXaDJCM2t6dmFkT2s4U2p5bnBEOEJBNXFnb2EifQ.c9b3w3RtkeEx5lAJLbqQ_VKsr4-aGPROIHOnJ1K7VAUKruI_nttTb72dTc5OsR9ecfCC9BQ7yQSE8XMoJNZRCVrZNonyOXTGQ2D4W4NlPVrKst3zl7RMjSTueocxks_En_zjcSAtGRvj6Fmsky9sX5hEOGx8Hx7UvuCrZk5QhdIU_h4Xj3c59GlKswGuNZIl6CX5XKwfRnSKDUmms08x4o8CgOCrVnipM7G8f4G2rXYxIPq_R3msQFl08kCvSzn6zHLVtFKpvi1kQU-NqCXqwKRxa_EhPHTfytNdGKpeE8xm866uyn8taklrhR4xKgLcZt1_495jffUiCLj7hAfLDA'
    }
    try:
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            models = [model["model"] for model in data.get("models", [])]
            return models
        else:
            return ["Error: Could not retrieve models - HTTP Status " + str(response.status_code)]
    except requests.exceptions.RequestException as e:
        return ["Error: Request failed - " + str(e)]

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
            index=models.index(st.session_state['selected_model'])
        )
        
        st.session_state['selected_model'] = selected_model
        logout()
    else:
        login()

    if 'user' in st.session_state:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Message LLM..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.spinner("Processing..."):
                # Call local LLM function
                chat_context = "\n".join([message["content"] for message in st.session_state.messages])
                prompt = prompt_template(instructions_string, chat_context)
                response, model_name = send_prompt_to_api(prompt, st.session_state['selected_model'])
                if response.startswith("Error:"):
                    st.error(response)

            with st.chat_message("assistant"):
                st.markdown(response)

            # Append assistant response to messages
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()
