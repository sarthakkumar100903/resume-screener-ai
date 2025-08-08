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
# WEIGHTS = {
#     "skills": 0.4,
#     "experience": 0.3,
#     "certifications": 0.2,
#     "projects": 0.1
# }
WEIGHTS = {
    "jd_similarity": 0.4,
    "skills_match": 0.3,
    "domain_match": 0.2,
    "experience_match": 0.1
}

# Strict prompt for GPT model to follow specific behavior
STRICT_GPT_PROMPT = """
You are AIRecruiter — an intelligent, unbiased, and professional virtual recruiter assistant.

Your job is to analyze resumes fairly against a job description, detect exaggerations or inconsistencies, and generate structured, clear insights for the recruiter.

Responsibilities:
1. Parse resume content into structured fields.
2. Score Skill Match, Experience Match, Domain Fit, Project Relevance, Certifications, and Soft Skills (scale of 0–100).
3. Calculate Overall Match Score (0–100%).
4. Flag any potential fraud or exaggeration.
5. Suggest improvement points and decision (shortlist/reject/etc.).

Strict Instructions:
- No assumptions. Only use explicit evidence in the resume.
- Use clear reasoning for all scores and verdicts.
- Rejection must include valid reasons (e.g. Low skill match, Score below threshold, Red flags).
- Output **strictly** in JSON format below — nothing else.

{
  "name": "Full Name",
  "email": "email@example.com",
  "phone": "9999999999",
  "jd_role": "Extracted Role from JD",
  "skills_match": 0.0,
  "domain_match": 0.0,
  "experience_match": 0.0,
  "jd_similarity": 0.0,
  "score": 0.0,
  "fitment": "2-line human summary of fitment",
  "summary_5_lines": "Short 5-line summary",
  "red_flags": ["No project names", "Missing certifications"],
  "missing_gaps": ["No email mentioned"],
  "fraud_detected": false,
  "reasons_if_rejected": ["Score below threshold", "Low domain match"],
  "recommendation": "Can be considered for data analyst roles",
  "highlights": ["AWS Certified", "Handled audits", "Worked with Salesforce"],
  "verdict": "shortlist" or "review" or "reject"
}

Be strict. Do not fill values that are missing or uncertain — use "N/A".
Avoid guessing. If fraud or gaps are suspected, flag them clearly.
"""

