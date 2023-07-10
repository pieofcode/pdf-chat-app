import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectors(chunks):
    if os.environ["OPENAI_CONFIG"] == "AZURE":
        embeddings = OpenAIEmbeddings(
            deployment=os.environ["AZ_EMBEDDING_DEPLOYMENT_NAME"],
            model=os.environ["AZ_EMBEDDING_DEPLOYMENT_NAME"],
            chunk_size=1,
            openai_api_key=os.environ["AZ_OPENAI_API_KEY"]
        )
    else:
        embeddings = OpenAIEmbeddings()

    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vector_store


def get_conversation_chain(vector_store):

    if os.environ["OPENAI_CONFIG"] == "AZURE":
        llm = AzureChatOpenAI(
            deployment_name=os.environ["AZ_CHATGPT_DEPLOYMENT_NAME"],
            openai_api_base=os.environ["AZ_OPENAI_API_BASE"],
            openai_api_type="azure",
            openai_api_version=os.environ["AZ_API_VERSION"],
            openai_api_key=os.environ["AZ_OPENAI_API_KEY"]
        )
    else:
        llm = ChatOpenAI()

    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )

    return conversation_chain
