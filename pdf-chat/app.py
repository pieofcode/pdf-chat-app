import os
import streamlit as st
from pathlib import Path
from openai import AzureOpenAI
import dotenv
from framework.text_loader import *
from framework.az_ai_search_helper import *
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.callbacks import get_openai_callback

from html_template import css, bot_template, user_template

env_name = os.environ["APP_ENV"] if "APP_ENV" in os.environ else "local"

# Load env settings
env_file_path = Path(f"./.env.{env_name}")
print(f"Loading environment from: {env_file_path}")
with open(env_file_path) as f:
    dotenv.load_dotenv(dotenv_path=env_file_path)
# print(os.environ)

openai.api_type: str = "azure"
openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
openai.api_base = os.getenv("AZURE_OPENAI_ENDPOINT")
openai.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
model: str = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")

openai_client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)


def handle_user_input(question):
    with get_openai_callback() as cb:
        if not st.session_state.has_vectorized_data:
            st.write(
                "Please upload your documents and hit Process to build vector store.")
            return

        response = st.session_state.conversation({"question": question})
        st.session_state.chat_history = response['chat_history']
        # st.write(response)
        # print(f"Chat History Type: {type(st.session_state.chat_history)}")
        for i, message in enumerate(reversed(st.session_state.chat_history)):
            print(F"Idx: {i}, Message: {message}")
            if type(message) == HumanMessage:
                st.write(user_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

            elif type(message) == AIMessage:
                st.write(bot_template.replace(
                    "{{MSG}}", message.content), unsafe_allow_html=True)

            else:
                st.write(
                    f"Error displaying message of Type[{type(message)}], Content[{message.content}]")

        print(cb)


def main():

    st.set_page_config(page_title="PDF Chatbot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize Session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "has_vectorized_data" not in st.session_state:
        st.session_state.has_vectorized_data = None

    if "use_az_search_vector_store" not in st.session_state:
        st.session_state.use_az_search_vector_store = None

    if "selected_index" not in st.session_state:
        st.session_state.selected_index = None

    st.header("Chat with your Data :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace(
        "{{MSG}}", "Hello Bot!"), unsafe_allow_html=True)
    st.write(bot_template.replace(
        "{{MSG}}", "Hello Human!"), unsafe_allow_html=True)

    with st.sidebar:

        on = st.toggle('Use Azure AI Search Vector Store', value=True)
        if on:
            if (st.session_state.use_az_search_vector_store == True):
                print("No action to be taken")
                st.stop()

            st.session_state.use_az_search_vector_store = True
            with st.spinner("Processing"):
                indices = get_az_search_indices()
                selected_index = st.selectbox(
                    'Choose Vector Index to use',
                    indices
                )
                # Step 3: Create embeddings and store in Vector store
                vector_store = get_az_search_vector_store(selected_index)

                # Step 4: Get conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vector_store=vector_store)

                st.write('You selected:', selected_index)
                st.session_state.selected_index = selected_index
                st.session_state.has_vectorized_data = True

        else:

            if ((st.session_state.use_az_search_vector_store == False) and
                    (st.session_state.has_vectorized_data == True)):
                print("No action to be taken")
                st.stop()

            st.session_state.use_az_search_vector_store = False
            st.subheader("Choose your knowledge base")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process' ", accept_multiple_files=True)
            if st.button("Process", type="primary"):

                if len(pdf_docs) != 0:

                    # process the information from PDFs
                    with st.spinner("Processing"):

                        # Step 1: Get raw contents from PDFs
                        raw_text = get_pdf_text(pdf_docs)

                        # Step 2: Get the chunks of the text
                        text_chunks = get_text_chunks(raw_text)
                        st.write(
                            f"Total length of the chunks: {len(text_chunks)}")
                        st.write(text_chunks)

                        # Step 3: Create embeddings and store in Vector store
                        vector_store = get_vectors(text_chunks)

                        # Step 4: Get conversation chain
                        st.session_state.conversation = get_conversation_chain(
                            vector_store=vector_store)

                        st.session_state.has_vectorized_data = True

        # add_sidebar = st.sidebar.selectbox(
        #     "EDSP Data Science", ('Data Engineering', 'Model Training'))


if __name__ == "__main__":
    main()
