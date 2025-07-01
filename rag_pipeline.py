import os
from dotenv import load_dotenv

from langchain_community.chat_models import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Load API key
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Load your WorkSafe PDF
loader = PyPDFLoader("docs/ladders.pdf")  # Update this if you use a different file
documents = loader.load()

# Split the PDF into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings and store in Chroma
embedding = OpenAIEmbeddings(openai_api_key=openai_key)
vectordb = Chroma.from_documents(docs, embedding, persist_directory="chromadb/")
vectordb.persist()

# Create retriever and QA chain
retriever = vectordb.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4", temperature=0.2, openai_api_key=openai_key),
    retriever=retriever
)

# Final callable function for the app
def get_response(query):
    return qa_chain.run(query)
