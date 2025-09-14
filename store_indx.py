from src.helper import load_pdf_files_from_directory, text_split, download_hugging_face_embeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Step 1: Load PDF documents
print(" Loading PDF files...")
extracted_data = load_pdf_files_from_directory("C:/clg/AI-Powered-Health-Companion/data")

# Step 2: Split into text chunks
print(" Splitting documents into chunks...")
text_chunks = text_split(extracted_data)

# Step 3: Load Hugging Face Embeddings
print(" Downloading embedding model...")
embedding_function = download_hugging_face_embeddings()

# Step 4: Initialize ChromaDB Client and store data
print(" Storing embeddings in ChromaDB...")
db = Chroma.from_documents(
    text_chunks,
    embedding_function,
    persist_directory="chroma_db/"
)

print(" All data has been embedded and stored successfully in ChromaDB.")
