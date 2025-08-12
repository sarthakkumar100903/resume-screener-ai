import os
from typing import Dict, Any

# Azure Configuration - Enhanced with better defaults
AZURE_CONFIG = {
    "openai_key": os.getenv("AZURE_OPENAI_KEY", "E8H5xVQpBp0bhrBW0ejunGDae8Xm17d6xnonrYXPhN1jf4G76OeMJQQJ99BHACfhMk5XJ3w3AAABACOGVpeS"),
    "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT", "https://screener-resume.openai.azure.com/"),
    "api_version": "2024-04-01-preview",
    "connection_string": os.getenv("AZURE_STORAGE_CONNECTION_STRING", 
                                 "DefaultEndpointsProtocol=https;AccountName=resumescreenerstorage;AccountKey=viOxwfuH/revazwvooYHEBfbVmLaPLdEzwIw6KVjK6exkopdQcLhHzP/WoLzXYIzR1kVEQilT/jP+ASt60C1gA==;EndpointSuffix=core.windows.net"),
    "resumes_container": "resumes",
    "summaries_container": "summaries", 
    "csv_container": "csvdata"
}

# Model Configuration - Optimized for performance
MODEL_CONFIG = {
    "fast_gpt_model": "gpt-35-turbo",           # For role extraction (faster)
    "deep_gpt_model": "gpt-4o",                 # For resume evaluation (more accurate)
    "embedding_model": "text-embedding-ada-002" # For similarity calculations
}

# Enhanced Scoring Weights - Fine-tuned for better results
WEIGHTS = {
    "jd_similarity": 0.35,      # Slightly reduced for more balanced scoring
    "skills_match": 0.35,       # Increased importance of skills
    "domain_match": 0.20,       # Domain relevance
    "experience_match": 0.10    # Experience level matching
}

# Default Thresholds - Optimized for three-tier system
DEFAULT_THRESHOLDS = {
    "shortlist_threshold": 75,   # Auto-shortlist above this score
    "reject_threshold": 40,      # Auto-reject below this score
    "jd_similarity_min": 60,     # Minimum JD similarity
    "skills_match_min": 65,      # Minimum skills match
    "domain_match_min": 50,      # Minimum domain match
    "experience_match_min": 55   # Minimum experience match
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "max_resume_chunks": 2,      # Reduced for faster processing
    "chunk_size": 1500,          # Optimal chunk size for GPT
    "chunk_overlap": 150,        # Overlap between chunks
    "batch_size": 5,             # Process 5 resumes concurrently
    "request_timeout": 30.0,     # Timeout for GPT requests
    "max_retries": 3,            # Retry failed requests
    "rate_limit_delay": 0.5      # Delay between batches (seconds)
}

# Enhanced GPT Prompt - Optimized for consistency and speed
STRICT_GPT_PROMPT = """
You are AIRecruiter — an expert, unbiased virtual recruiter that analyzes resumes against job descriptions.

CORE RESPONSIBILITIES:
1. Extract candidate information accurately
2. Score compatibility across multiple dimensions (0-100 scale)
3. Identify potential issues or red flags
4. Provide clear, actionable insights
5. Make hiring recommendations

SCORING CRITERIA (0-100):
- skills_match: How well candidate's skills align with job requirements
- domain_match: Relevance of candidate's industry/domain experience  
- experience_match: Compatibility of experience level and type
- jd_similarity: Overall alignment with job description

ANALYSIS RULES:
- Be objective and evidence-based
- Use only information explicitly stated in resume
- Flag inconsistencies, gaps, or exaggerations
- Consider cultural fit and growth potential
- Provide specific examples to support scores

VERDICT OPTIONS:
- "shortlist": Strong match, recommend for interview
- "review": Potential match, needs closer evaluation
- "reject": Poor match or significant concerns

OUTPUT FORMAT (JSON only):
{
  "name": "Full Name",
  "email": "email@domain.com", 
  "phone": "+1234567890",
  "jd_role": "Role from JD",
  "skills_match": 75,
  "domain_match": 80,
  "experience_match": 70,
  "fitment": "Concise 2-sentence fit summary",
  "summary_5_lines": "Brief candidate overview highlighting key strengths",
  "red_flags": ["Specific concerns if any"],
  "missing_gaps": ["Information gaps or missing details"],
  "fraud_detected": false,
  "reasons_if_rejected": ["Specific rejection reasons"],
  "recommendation": "Alternative role suggestion or next steps",
  "highlights": ["Key strengths and achievements"],
  "verdict": "shortlist|review|reject"
}

CRITICAL REQUIREMENTS:
- Output ONLY valid JSON, no additional text
- Use "N/A" for missing information, never guess
- Be consistent in scoring methodology
- Flag suspicious patterns or inconsistencies
- Justify low scores with specific reasons
"""

# Email Templates - Enhanced for better communication
EMAIL_TEMPLATES = {
    "shortlist": {
        "subject": "Congratulations! You've been shortlisted for {role}",
        "body": """Dear {name},

Congratulations! After reviewing your application for the {role} position, we are pleased to inform you that you have been shortlisted for the next round of our selection process.

Our recruitment team was impressed with your qualifications and experience, particularly:
{highlights}

We will be in touch soon with details about the next steps, including interview scheduling and additional requirements.

Thank you for your interest in our organization, and we look forward to speaking with you soon.

Best regards,
{company_name} Recruitment Team"""
    },
    
    "review": {
        "subject": "Application Update - Additional Review Required",
        "body": """Dear {name},

Thank you for your application for the {role} position.

Your profile is currently under review by our recruitment team. We may need some additional information or clarification to proceed with your application.

We will be in touch within the next few business days with any questions or requests for additional documentation.

Thank you for your patience during this process.

Best regards,  
{company_name} Recruitment Team"""
    },
    
    "reject": {
        "subject": "Application Status Update - {role} Position",
        "body": """Dear {name},

Thank you for your interest in the {role} position at {company_name}. 

After careful consideration of all applications, we have decided not to proceed with your application at this time. This decision does not reflect on your qualifications, as we received many strong applications for this position.

We appreciate the time and effort you invested in the application process and encourage you to apply for future opportunities that may be a better match for your background.

We wish you success in your career endeavors.

Best regards,
{company_name} Recruitment Team"""
    }
}

# Fraud Detection Patterns
FRAUD_PATTERNS = {
    "suspicious_keywords": [
        "photoshop", "fake", "modified", "edited", "template", 
        "generated", "sample", "example", "dummy", "placeholder"
    ],
    "red_flag_phrases": [
        "responsible for everything", "expert in all technologies", 
        "single-handedly", "revolutionized the company", "increased revenue by 1000%",
        "worked with all Fortune 500", "invented", "pioneered"
    ],
    "missing_info_flags": [
        "no contact information", "no email", "no phone",
        "no company names", "no dates", "vague descriptions",
        "no specific achievements", "generic responsibilities"
    ]
}

# Quality Control Settings
QUALITY_CONTROL = {
    "min_resume_length": 100,        # Minimum characters in resume
    "max_resume_length": 50000,      # Maximum characters in resume  
    "min_score_variance": 10,        # Minimum variance in scoring
    "max_processing_time": 45.0,     # Maximum time per resume (seconds)
    "required_contact_fields": ["name", "email"],  # Required contact info
    "suspicious_score_threshold": 95  # Flag perfect scores as suspicious
}

# Analytics Configuration
ANALYTICS_CONFIG = {
    "score_bins": [0, 20, 40, 60, 80, 100],
    "score_labels": ["Poor", "Below Average", "Average", "Good", "Excellent"],
    "performance_metrics": [
        "processing_time", "success_rate", "average_score",
        "shortlist_rate", "rejection_rate", "review_rate"
    ]
}

# Export Settings
EXPORT_CONFIG = {
    "csv_columns": [
        "name", "email", "phone", "jd_role", "score", "verdict",
        "skills_match", "domain_match", "experience_match", "jd_similarity",
        "fitment", "highlights", "red_flags", "recruiter_notes"
    ],
    "summary_fields": [
        "name", "email", "phone", "score", "verdict", "fitment",
        "summary_5_lines", "highlights", "red_flags", "recommendation"
    ]
}

# Validation Schemas
VALIDATION_SCHEMAS = {
    "candidate_response": {
        "required_fields": ["name", "verdict", "score"],
        "score_fields": ["skills_match", "domain_match", "experience_match"],
        "list_fields": ["red_flags", "missing_gaps", "highlights", "reasons_if_rejected"],
        "text_fields": ["fitment", "summary_5_lines", "recommendation"]
    }
}

# System Messages and Prompts
SYSTEM_MESSAGES = {
    "processing_start": "🚀 Starting AI-powered resume analysis...",
    "processing_complete": "✅ Analysis completed successfully!",
    "error_processing": "❌ Error during processing: {error}",
    "no_resumes": "⚠️ No resumes found to process",
    "performance_summary": "📊 Processed {count} resumes in {time:.2f}s (avg: {avg:.2f}s per resume)"
}

# Feature Flags - Enable/disable features
FEATURE_FLAGS = {
    "enable_fraud_detection": True,
    "enable_performance_monitoring": True,
    "enable_batch_processing": True,
    "enable_caching": True,
    "enable_detailed_logging": True,
    "enable_auto_email": True,
    "enable_pdf_generation": True,
    "enable_blob_storage": True
}

# Development/Debug Settings
DEBUG_CONFIG = {
    "log_level": "INFO",
    "log_gpt_requests": False,      # Set to True for debugging GPT calls
    "log_performance": True,        # Log performance metrics
    "save_failed_responses": True,  # Save failed GPT responses for debugging
    "mock_gpt_responses": False     # Use mock responses for testing
}
