import streamlit as st
from dotenv import load_dotenv
load_dotenv()

import os
import google.generativeai as genai

from langchain.schema import Document
from PyPDF2 import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS

#API Configuration
genai.configure(api_key = os.getenv("Google-API-KEY"))
model = genai.GenerativeModel('gemini-2.0-flash')

#Cache the HF Embeddings to avoid slow reload of the embeddings
@st.cache_resource(show_spinner="Loading Embedding Model")
def embeddings():
    return(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
embedding_model = embeddings()

#User Interface
st.header("RAG Assistant: :orange[HF Embeddings +Gemini LLM]")
st.subheader("Your AI Doc Assistant")
uploaded_file = st.file_uploader(label="Upload the PDF Doc",type=['PDF'])

if uploaded_file:
    raw_text=""
    pdf = PdfReader(uploaded_file)
    for i,page in enumerate(pdf.pages):
        text = page.extract_text()
        if text:
            raw_text+=text
    if raw_text.strip():
        document = Document(page_content=raw_text)
        #using charactertextsplitter we will create chunks and pass it into model
        splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        chunks = splitter.split_documents([document])
        #store the chunks into FAISS vectordb
        chunk_pieces = [chunk.page_content for chunk in chunks]
        vectordb = FAISS.from_texts(chunk_pieces,embedding_model) #convert them ino vectors 
        retriever = vectordb.as_retriever() #Retrieve the vectors
        st.success("Embeddings are Generated.Ask Your Question!!")
        
        #User Q&A
        user_input = st.text_input(label = "Enter your Question:")
        if user_input:
            with st.chat_message("user"):
                st.write(user_input)
                
            with st.spinner("Analyzing the documents..."):
                relevant_docs = retriever.get_relevant_documents(user_input)
                context = '\n\n'.join(doc.page_content for doc in relevant_docs)
                
                prompt = f''' You are an expert assistant and use the context below to answer the query
                If unsure or information not available in the doc, pass the message-"Information is not Available.Look into Other Sources"
                context = {context},
                query = {user_input},
                Answer:'''
                response = model.generate_content(prompt)
                st.markdown("Answer:")
                st.write(response.text)
    else:
        st.warning("No Text could be Extracted from PDF.Please Upload A Readable PDF")
            
            
            