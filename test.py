"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

# from langchain.chains import ConversationChain
# from langchain.llms import OpenAI

from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate

st.write('### Loading book')
loader = UnstructuredPDFLoader("book2.pdf")
data = loader.load()

# print (f'You have {len(data)} document(s) in your data')
st.write(f'You have {len(data)} document(s) in your data')

st.write(f'There are {len(data[0].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

st.write(f'Now you have {len(texts)} documents')

st.write(f'Example document: {texts[50]}')


st.write('### Create embeddings of your documents to get ready for semantic search')
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

PINECONE_API_KEY = '123deabc-0a96-41b8-b668-77a5e1d0a437'
PINECONE_API_ENV = 'us-east4-gcp'

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
st.write('Embeddings created:')
st.write(embeddings)
