import imaplib
import email
from email.header import decode_header
from azure.storage.blob import BlobServiceClient

# Gmail credentials
EMAIL = "demoprojectid3@gmail.com"
PASSWORD = "qthzxwpsqhlslcxj"  # Your App Password (no spaces)
IMAP_SERVER = "imap.gmail.com"
IMAP_PORT = 993

# Azure credentials
AZURE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;AccountName=resumescreenerst;AccountKey=H8sGhn9NpR5qoDrTJpdXZxGpBM3h67hChtd4B4v7vIy8QG3lv8cNIdUvnBoTDwyvN3YhtQH56Hbr+AStrMNVbA==;EndpointSuffix=core.windows.net"
CONTAINER_NAME = "resumes"

# Connect to Gmail
mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)
mail.login(EMAIL, PASSWORD)
mail.select("inbox")

# Search unread emails
status, messages = mail.search(None, '(UNSEEN)')
email_ids = messages[0].split()

print(f"📨 Found {len(email_ids)} unread emails.")

# Connect to Azure Blob Storage
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME)

# Create container if it doesn’t exist
try:
    container_client.create_container()
except Exception:
    pass  # already exists

# Loop through emails
for e_id in email_ids:
    _, msg_data = mail.fetch(e_id, "(RFC822)")
    raw_email = msg_data[0][1]
    msg = email.message_from_bytes(raw_email)

    subject, encoding = decode_header(msg["Subject"])[0]
    if isinstance(subject, bytes):
        subject = subject.decode(encoding or "utf-8")
    print(f"📩 Processing email: {subject}")

    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        if part.get("Content-Disposition") is None:
            continue

        filename = part.get_filename()
        if filename:
            filename = decode_header(filename)[0][0]
            if isinstance(filename, bytes):
                filename = filename.decode()

            payload = part.get_payload(decode=True)
            blob_client = container_client.get_blob_client(filename)
            blob_client.upload_blob(payload, overwrite=True)
            print(f"✅ Uploaded '{filename}' to Azure Blob Storage")

    # Mark email as read
    mail.store(e_id, "+FLAGS", "\\Seen")

mail.logout()


