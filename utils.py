import re
import fitz  # PyMuPDF
import numpy as np
import tiktoken
import functools
import os
from azure.storage.blob import BlobServiceClient
from sklearn.metrics.pairwise import cosine_similarity
from constants import AZURE_CONFIG, MODEL_CONFIG
from openai import AzureOpenAI

# ==========================
# 📄 Resume Text Extractor
# ==========================
def parse_resume(file_bytes):
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = ""
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"❌ Error parsing resume: {e}")
        return "Error reading resume"

# ==========================
# 📎 Chunk Text for GPT or Embedding
# ==========================
def chunk_text(text, max_tokens=3000, overlap=200):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(enc.decode(chunk))
        i += max_tokens - overlap
    return chunks

def get_text_chunks(text, max_tokens=800, overlap=100):
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(text)
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i + max_tokens]
        chunks.append(enc.decode(chunk))
        i += max_tokens - overlap
    return chunks

# ==========================
# 🧠 Embedding + Similarity
# ==========================
def get_embedding(text):
    client = AzureOpenAI(
        api_key=AZURE_CONFIG["openai_key"],
        api_version=AZURE_CONFIG["api_version"],
        azure_endpoint=AZURE_CONFIG["azure_endpoint"]
    )
    try:
        response = client.embeddings.create(
            input=[text],
            model=MODEL_CONFIG["embedding_model"]
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"❌ Error generating embedding: {e}")
        return [0.0] * 1536  # fallback vector

@functools.lru_cache(maxsize=10)
def get_embedding_cached(text):
    return tuple(get_embedding(text))

def get_cosine_similarity(vec1, vec2):
    try:
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0
        return cosine_similarity([vec1], [vec2])[0][0]
    except Exception as e:
        print(f"❌ Error in cosine similarity: {e}")
        return 0.0

# ==========================
# 📬 Contact Info Extractor with Name Fallback
# ==========================
def extract_contact_info(text):
    name = "N/A"
    email_match = re.search(r"[\w\.-]+@[\w\.-]+", text)
    phone_match = re.search(r"\+?\d[\d\s\-]{8,}", text)

    email = email_match.group(0) if email_match else "N/A"
    phone = phone_match.group(0).replace(" ", "").replace("-", "") if phone_match else "N/A"

    lines = text.splitlines()
    for line in lines[:10]:
        line_clean = line.strip()
        if len(line_clean.split()) <= 5 and any(
            kw.lower() in line_clean.lower() for kw in ["name", "curriculum", "cv", "resume"]
        ):
            name = re.sub(r"(name|cv|resume|curriculum vitae)[:\-]?", "", line_clean, flags=re.I).strip()
            break

    if name == "N/A":
        for line in lines[:10]:
            if line.strip() and line.strip()[0].isupper() and len(line.strip().split()) <= 5:
                name = line.strip()
                break

    return {
        "name": name or "N/A",
        "email": email,
        "phone": phone
    }

# ==========================
# ☁️ Azure Uploads
# ==========================
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

def upload_to_blob(file_bytes, file_name, container):
    try:
        container_client = blob_service_client.get_container_client(container)
        blob_client = container_client.get_blob_client(file_name)
        blob_client.upload_blob(file_bytes, overwrite=True)
    except Exception as e:
        print(f"❌ Upload to blob failed: {e}")

def save_summary_to_blob(pdf_bytes, file_name, container):
    try:
        container_client = blob_service_client.get_container_client(container)
        blob_client = container_client.get_blob_client(file_name)
        blob_client.upload_blob(pdf_bytes, overwrite=True)
    except Exception as e:
        print(f"❌ Saving summary failed: {e}")

def save_csv_to_blob(df, file_name, container):
    try:
        container_client = blob_service_client.get_container_client(container)
        blob_client = container_client.get_blob_client(file_name)
        blob_client.upload_blob(df.to_csv(index=False), overwrite=True)
    except Exception as e:
        print(f"❌ Saving CSV failed: {e}")
