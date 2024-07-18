import streamlit as st
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from set_env import set_env

# Set environment variables
set_env()

# Streamlit app
st.title("CSV Dataframe Q&A with LangChain")

# Initialize chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Upload CSV file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Dataframe loaded:")
    st.write(df)

    # Initialize LangChain agent
    agent = create_pandas_dataframe_agent(
        ChatOpenAI(openai_api_base="http://199.204.135.71:11434/v1", model="mistral", openai_api_key="ollama"),
        df,
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        handle_parsing_errors=True,
    )

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
        user_question = st.text_input("Ask a question about the data:")
        submit_button = st.form_submit_button(label='Send')

        if submit_button and user_question:
            st.session_state.messages.append({"role": "user", "content": user_question})

            with st.spinner("Processing..."):
                try:
                    response = agent.invoke(user_question)
                    # Debugging: print the response to the console
                    print(response)
                    # Handle response based on expected structure
                    if isinstance(response, dict) and 'output' in response:
                        response_content = response['output']
                    else:
                        response_content = response if isinstance(response, str) else str(response)
                    st.session_state.messages.append({"role": "assistant", "content": response_content})
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

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

else:
    st.write("Please upload a CSV file to proceed.")
