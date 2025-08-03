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
MODEL_CONFIG = {
    "model_name": "gpt-35-turbo",  # or "gpt-4" based on your usage
    "temperature": 0.7,
    "max_tokens": 1000
}

# Weights for scoring (you can customize these as needed)
WEIGHTS = {
    "skills": 0.4,
    "experience": 0.3,
    "certifications": 0.2,
    "projects": 0.1
}

# Strict prompt for GPT model to follow specific behavior
STRICT_GPT_PROMPT = """
You are an AI assistant that evaluates resumes strictly based on job description criteria.
Return JSON format:
{
  "match_score": <0 to 100>,
  "matched_skills": [...],
  "missing_skills": [...],
  "summary": "<brief summary of fit>"
}
Do not return anything else.
"""
