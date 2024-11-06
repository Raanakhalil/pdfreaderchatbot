import os
import fitz  # PyMuPDF for PDF processing
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from groq import Groq

# Initialize the Groq API client
client = Groq(api_key="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")

# Load and split PDF content into chunks
def load_pdf_to_text(pdf_path):
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return docs

# Initialize vector store with Hugging Face embeddings (open-source model)
def create_vector_store(docs):
    embeddings = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Query the vector store and get relevant chunks
def get_relevant_chunks(vector_store, query):
    docs = vector_store.similarity_search(query, k=5)
    return " ".join([doc.page_content for doc in docs])

# Generate response using Groq API
def generate_answer(query, context):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": f"Using the following information: {context}. Answer the question: {query}"}
        ],
        model="llama3-8b-8192",
    )
    return chat_completion.choices[0].message.content

# Main function for PDF QA
def pdf_qa_bot(pdf_path, question):
    docs = load_pdf_to_text(pdf_path)
    vector_store = create_vector_store(docs)
    context = get_relevant_chunks(vector_store, question)
    answer = generate_answer(question, context)
    return answer

