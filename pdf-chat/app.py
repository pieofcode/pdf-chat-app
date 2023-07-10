import os
import streamlit as st
from pathlib import Path
import openai
import dotenv

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

from langchain.callbacks import get_openai_callback

from html_template import css, bot_template, user_template
from framework.text_loader import *

env_name = os.environ["APP_ENV"] if "APP_ENV" in os.environ else "local"

# Load env settings
env_file_path = Path(f"./.env.{env_name}")
dotenv.load_dotenv(dotenv_path=env_file_path)
# print(os.environ)


if os.environ["OPENAI_CONFIG"] == "AZURE":
    openai.api_base = os.environ["AZ_OPENAI_API_BASE"]
    openai.api_type = "azure"
    openai.api_version = os.environ["AZ_API_VERSION"]
    openai.api_key = os.environ["AZ_OPENAI_API_KEY"]


def handle_user_input(question):
    with get_openai_callback() as cb:
        if not st.session_state.has_vectorized_data:
            st.write(
                "Please upload your documents and hit Process to build vector store.")
            return

        response = st.session_state.conversation({"question": question})
        st.session_state.chat_history = response['chat_history']
        # st.write(response)
        print(f"Chat History Type: {type(st.session_state.chat_history)}")
        for i, message in enumerate(reversed(st.session_state.chat_history)):

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


def init_az_openai_env():
    global openai

    openai.api_base = os.environ["AZ_OPENAI_API_BASE"]
    openai.api_type = "azure"
    openai.api_version = os.environ["AZ_API_VERSION"]
    openai.api_key = os.environ["AZ_OPENAI_API_KEY"]

    # https://github.com/hwchase17/langchain/issues/2096
    # https://github.com/hwchase17/langchain/issues/4575
    # This issue prevents multiple chunks to be added using Azure Text Embedding models


def init_openai_env():
    pass


def main():

    st.set_page_config("PDF Chatbot", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    # Initialize Session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    if "has_vectorized_data" not in st.session_state:
        st.session_state.has_vectorized_data = None

    st.header("Chat with your Data :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    st.write(user_template.replace(
        "{{MSG}}", "Hello Bot!"), unsafe_allow_html=True)
    st.write(bot_template.replace(
        "{{MSG}}", "Hello Human!"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
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
                    st.write(f"Total length of the chunks: {len(text_chunks)}")
                    st.write(text_chunks)

                    # Step 3: Create embeddings and store in Vector store
                    vector_store = get_vectors(text_chunks)

                    # Step 4: Get conversation chain
                    st.session_state.conversation = get_conversation_chain(
                        vector_store=vector_store)

                    st.session_state.has_vectorized_data = True


if __name__ == "__main__":
    main()
