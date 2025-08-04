# ==========================
# ☁️ Azure Uploads (Resumes, PDFs, CSVs)
# ==========================
import os
from azure.storage.blob import BlobServiceClient

# Read connection string directly from environment (via .env or GitHub Secret)
connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

# Create global blob service client
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
