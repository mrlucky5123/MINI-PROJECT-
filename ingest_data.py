import os
import time
import chromadb
import cohere
import PyPDF2
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from cohere.errors.too_many_requests_error import TooManyRequestsError

# --- 1. Load API keys and connect to Cohere ---
load_dotenv()
api_key = os.getenv("COHERE_API_KEY")
if api_key:
    api_key = api_key.strip()
else:
    print("Error: COHERE_API_KEY not found in .env file.")
    exit()

co = cohere.Client(api_key)

def extract_text_from_pdf(pdf_file_path):
    """
    Extracts raw text from a PDF file using PyPDF2.
    """
    text = ""
    with open(pdf_file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text

# --- Step 1: Extract text from PDF ---
pdf_path = "Competitive Programmer's Handbook.pdf"
if not os.path.exists(pdf_path):
    print(f"Error: '{pdf_path}' not found in the current directory.")
    print("Please make sure the PDF file is in the same folder as this script.")
    exit()

print("Extracting text from PDF...")
pdf_text = extract_text_from_pdf(pdf_path)

# --- Step 2: Split the text into production-grade chunks ---
# Expanded chunk size to 1500 characters (~375 tokens) to protect complex code syntax structures
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1800,   
    chunk_overlap=400   # Overlap increased to maintain contextual links across partitions
)
docs = text_splitter.create_documents([pdf_text])
chunks = [doc.page_content for doc in docs]

# --- Step 3: Initialize Persistent Vector Database ---
client = chromadb.PersistentClient(path="./chroma_db")

try:
    client.delete_collection(name="rag_collection_pdfs")
    print("Old collection wiped successfully.")
except Exception:
    pass 

collection = client.get_or_create_collection(name="rag_collection_pdfs")

# --- Step 4: Optimized Batch Ingestion with Rate-Limit Safeguards ---
BATCH_SIZE = 90
total_chunks = len(chunks)
print(f"Ingesting {total_chunks} larger context chunks into ChromaDB...")

for i in range(0, total_chunks, BATCH_SIZE):
    batch_chunks = chunks[i:i + BATCH_SIZE]
    batch_ids = [f"doc_{j}" for j in range(i, min(i + BATCH_SIZE, total_chunks))]
    
    while True:
        try:
            response = co.embed(
                texts=batch_chunks,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            batch_embeddings = response.embeddings

            collection.add(
                ids=batch_ids,
                documents=batch_chunks,
                embeddings=batch_embeddings
            )
            print(f"Successfully processed and ingested chunks {i} to {min(i + BATCH_SIZE, total_chunks)} / {total_chunks}")
            
            # 2-second rate control buffer
            time.sleep(2) 
            break 

        except TooManyRequestsError:
            print("\n[Rate Limit Triggered] Throttled by endpoint. Pausing for 15 seconds to flush limits...")
            time.sleep(15)
        except Exception as e:
            print(f"Unexpected error encountered: {e}")
            exit()

print("\nIngestion complete. ChromaDB collection 'rag_collection_pdfs' is optimized and ready on disk!")