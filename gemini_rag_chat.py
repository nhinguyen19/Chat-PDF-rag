import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import tempfile
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from helper_func import Chatbot

load_dotenv()

bot = Chatbot()

api_key= os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Hong co key roi")
    st.stop()

genai.configure(api_key=api_key)

st.set_page_config(page_title="Chat PDF RAG")
st.title("Chat bot Analyze Data")

user_question = st.text_input("Please asking chatbot after uploading files")

if user_question:
    answer = bot.user_input(user_question)
    st.write(answer)


with st.sidebar:
    st.title("Upload Files")
    pdf_docs = st.file_uploader("Uploading now", accept_multiple_files=True, type=["pdf"])

    if st.button("Analyze"):
        if not pdf_docs:
            st.error("Please uploading file before analyzing")
        else:
            with st.spinner("Procesing......................."):
                raw_text = bot.get_pdf_text(pdf_docs)
                if raw_text:
                    text_chunks = bot.get_text_chunk(raw_text)
                    if text_chunks:
                        bot.get_vector_store(text_chunks)
                    else:
                        st.error("Plese check the content in PDF files")


