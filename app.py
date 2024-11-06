import os
import fitz  # PyMuPDF for PDF processing
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
from groq import Groq
from tempfile import NamedTemporaryFile

# Set up the Groq client with your API key
client = Groq(api_key="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")

# Step 1: Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Step 2: Function to split extracted text into chunks for retrieval
def chunk_text(text, chunk_size=1000):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Step 3: Retrieve the most relevant chunk using TF-IDF and cosine similarity
def retrieve_chunk(question, chunks):
    vectorizer = TfidfVectorizer().fit_transform([question] + chunks)
    question_vector = vectorizer[0]
    chunk_vectors = vectorizer[1:]
    similarities = cosine_similarity(question_vector, chunk_vectors).flatten()
    best_chunk_index = np.argmax(similarities)
    return chunks[best_chunk_index]

# Step 4: Generate an answer using the Groq API's language model
def generate_answer(retrieved_text, question):
    prompt = f"Based on the following text, answer the question:\n\nText: {retrieved_text}\n\nQuestion: {question}"
    chat_completion = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"
    )
    return chat_completion.choices[0].message.content

# Step 5: Streamlit UI for PDF upload and Q&A
def main():
    st.title("PDF Question-Answer Chatbot")

    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file:
        with NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            pdf_path = tmp_file.name

        # Extract text from the uploaded PDF and chunk it
        text = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(text)

        question = st.text_input("Ask a question:")
        if st.button("Get Answer"):
            if question:
                retrieved_text = retrieve_chunk(question, chunks)
                answer = generate_answer(retrieved_text, question)
                st.write("Answer:", answer)
            else:
                st.write("Please enter a question.")

if __name__ == "__main__":
    main()
