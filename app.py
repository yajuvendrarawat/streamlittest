import streamlit as st

def main():
    load_dotenv()
    # Initialize session state
    initialize_session_state()
    st.title("Multi-Docs ChatBot using llama2 :books:")
    # Initialize Streamlit
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)    

if __name__ == "__main__":
    main()
