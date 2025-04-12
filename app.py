import time
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv
load_dotenv()

# load the groq and google api from .env file
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
llm

prompt = ChatPromptTemplate.from_template("""
Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in 
provided context just say, "answer is not available in the context", don't provide the wrong answer \n\n
<context>  {context} <context>
Questions: {input} 
""")


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001")
        st.session_state.loader = PyPDFDirectoryLoader(
            "./us_census")  # data ingestion
        st.session_state.docs = st.session_state.loader.load()  # docs loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents, st.session_state.embeddings)


prompt1 = st.text_input("What you want to ask from documents ?")
if st.button("Create vector store"):
    vector_embedding()
    st.write("Vector store DB is ready")


if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please create the vector store first by clicking the 'Create vector store' button.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write(response['answer'])

        with st.expander("Document similarity search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("---------------")