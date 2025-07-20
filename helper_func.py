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
import asyncio

class Chatbot:
    def __init__(self):
        load_dotenv() 

    def get_pdf_text(self, pdf_docs):
        text = ""
        try:
            for pdf in pdf_docs:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf.read())
                    tmp_file_path = tmp_file.name

                pdf_reader = PyPDFLoader(tmp_file_path)
                for page in pdf_reader.load_and_split():
                    text += page.page_content

                os.unlink(tmp_file_path)

        except Exception as e:
            st.error(f"Error in loading PDF files: {e}")
        
        return text

    def get_text_chunk(self, text):
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
            chunks = text_splitter.split_text(text)
            return chunks
        except Exception as e:
            st.error(f"Error in text splitting: {e}")
            return []

    def get_vector_store(self, text_chunks):
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))
            vector_store = FAISS.from_texts(text_chunks, embeddings)
            vector_store.save_local("faiss_index")
            st.success("Your files have been processed. Start asking something!")
        except Exception as e:
            st.error(f"Error in saving vector store: {e}")

    def get_conversational_chain(self):
        prompt_template = """
            Trả lời câu hỏi một cách chi tiết nhất có thể dựa trên ngữ cảnh được cung cấp. Nếu câu trả lời không có trong ngữ cảnh được cung cấp, hãy nói, "Câu trả lời không có trong ngữ cảnh."
                Không cung cấp thông tin sai lệch.

                Ngữ cảnh: {context}
                Câu hỏi: {question}

                Answer:
            """ 
        try:
            model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3) #temp: thông số độ sáng tạo của model
            prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"]) #Cách trả lời
            chain = load_qa_chain(model,chain_type="stuff", prompt=prompt)

            return chain
        except Exception as e:
            st.error(f"Error in analyze procees {str(e)}")

        return None

    def user_input(self,user_question):
        try: 
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.set_event_loop(asyncio.new_event_loop())

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=os.getenv("GOOGLE_API_KEY"))

            if not os.path.exists("faiss_index"):
                st.error("Cannot find FAISS INDEX file. Please upload file")
                return
            
            new_db = FAISS.load_local("faiss_index",embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            chain = self.get_conversational_chain()

            if not chain:
                return
            
            response = chain(
                {"input_documents": docs, "question": user_question},
                return_only_outputs=True)

        except Exception as e:
            st.error(f"Something wrong: {e}")
    
        return response["output_text"]