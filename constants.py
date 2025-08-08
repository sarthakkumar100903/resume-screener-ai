import os

AZURE_CONFIG = {
    "openai_key": "E8H5xVQpBp0bhrBW0ejunGDae8Xm17d6xnonrYXPhN1jf4G76OeMJQQJ99BHACfhMk5XJ3w3AAABACOGVpeS",
    "azure_endpoint": "https://screener-resume.openai.azure.com/",
    "api_version": "2024-04-01-preview",
    "connection_string": "DefaultEndpointsProtocol=https;AccountName=resumescreenerstorage;AccountKey=viOxwfuH/revazwvooYHEBfbVmLaPLdEzwIw6KVjK6exkopdQcLhHzP/WoLzXYIzR1kVEQilT/jP+ASt60C1gA==;EndpointSuffix=core.windows.net",
    "resumes_container": "resumes",
    "summaries_container": "summaries",
    "csv_container": "csvdata"
    # "openai_key": os.getenv("AZURE_OPENAI_KEY"),
    # "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
    # "api_version": "2024-04-01-preview",
    # "connection_string": os.getenv("AZURE_STORAGE_CONNECTION_STRING"),
    # "resumes_container": "resumes",
    # "summaries_container": "summaries",
    # "csv_container": "csvdata"
}
MODEL_CONFIG = {
    # "model_name": "gpt-35-turbo",  # or "gpt-4" based on your usage
    "fast_gpt_model": "gpt-35-turbo",             # JD role extraction
    "deep_gpt_model": "gpt-4.1",                  # Resume evaluation
    "embedding_model": "text-embedding-ada-002"   # JD-resume similarity
    # "temperature": 0.7,
    # "max_tokens": 1000
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
