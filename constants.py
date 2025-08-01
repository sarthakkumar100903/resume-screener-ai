import os

AZURE_CONFIG = {
    "openai_key": os.getenv("AZURE_OPENAI_KEY"),
    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    "api_version": "2024-04-01-preview",
    "connection_string": os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    "resumes_container": "resumes",
    "summaries_container": "summaries",
    "csv_container": "csvdata"
}
