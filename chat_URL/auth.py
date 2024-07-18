import streamlit as st

def login(auth):
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
