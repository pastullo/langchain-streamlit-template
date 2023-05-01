"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

# from langchain.chains import ConversationChain
# from langchain.llms import OpenAI

from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

st.write('Loading book')
loader = UnstructuredPDFLoader("book2.pdf")
data = loader.load()

# print (f'You have {len(data)} document(s) in your data')
st.write('You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')
st.write(f'There are {len(data[0].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
