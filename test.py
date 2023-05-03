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

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV = 'us-east4-gcp'

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
st.write('Embeddings created:')
st.write(embeddings)

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "test-index"

docsearch = Pinecone.from_texts(
    [t.page_content for t in texts], 
    embeddings, 
    index_name=index_name, 
    metadatas=[{"source": f"{t}-pl"} for t in texts]
)
print (f'{docsearch}')


# query = "What are the Three  Assurances of the cunnilingus?"
query = st.text_area("Your question", value="")

if st.button("Submit"):
    st.write("You asked:", query)

docs = docsearch.similarity_search(query, include_metadata=True)
print (f'Found {len(docs)} relevant excerpts.')

from langchain.llms import OpenAI
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

from langchain.chains.question_answering import load_qa_chain
chain = load_qa_chain(llm, chain_type="stuff")
chain.run(input_documents=docs, question=query)
