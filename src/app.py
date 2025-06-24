import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from dotenv import load_dotenv
import os

load_dotenv()

def get_vectorstore(website_url):
    loader = WebBaseLoader(website_url)
    documents = loader.load()

    embedding_model = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
    document_chunks = text_splitter.split_documents(documents)

    vector_store = Chroma.from_documents(document_chunks, embedding_model)
    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    rag_chain = get_conversational_rag_chain(retriever_chain)

    response = rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_input
    })
    
    return response["answer"]


# Streamlit App
st.set_page_config(page_title="ConvoWeb", page_icon="🤖")
st.title("🤖 ConvoWeb")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hello! I am a bot. How can I help you?")
    ]

with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if not website_url:
    st.info("Please enter a website URL to start.")
else:
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore(website_url)

    user_query = st.chat_input("Type your message here...")

    if user_query:
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
