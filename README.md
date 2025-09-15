# 🚀 EazyAI Resume Screener

**AI-powered resume screening platform that automates HR recruitment process**

## 🎯 What it does
- Reads PDF resumes automatically
- Compares them with job requirements using AI
- Scores each candidate (0-100%)
- Sorts into: ✅ Shortlist, 🟨 Review, ❌ Reject
- Sends automated emails to candidates
- Generates PDF reports

## ⚡ Key Benefits
- **95% time savings**: Process 100 resumes in 5 minutes instead of 8 hours
- **Smart AI analysis**: Uses GPT-4 to understand resume content
- **Bulk processing**: Handle multiple resumes simultaneously
- **Professional reports**: Generate PDF summaries for each candidate

## 🛠️ Technology Stack
- **Frontend**: Streamlit (Python web framework)
- **AI Engine**: Azure OpenAI (GPT-4 + Embeddings)
- **Storage**: Azure Blob Storage
- **Email**: Gmail SMTP integration
- **Reports**: PDF generation with ReportLab

## 📋 Prerequisites
- Python 3.8+
- Azure subscription (with OpenAI access)
- Gmail account for emails

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/yourusername/eazyai-resume-screener.git
cd eazyai-resume-screener
pip install -r requirements.txt
```

### 2. Configuration
Create `.env` file:
```bash
AZURE_OPENAI_KEY=your_openai_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
AZURE_STORAGE_CONNECTION_STRING="your_storage_connection"
HR_EMAIL=your-email@gmail.com
SMTP_PASS=your_gmail_app_password
```

### 3. Run Application
```bash
streamlit run app.py
```
Open http://localhost:8501 in your browser

## 💡 How to Use

1. **Setup Job**: Paste job description, set requirements
2. **Upload Resumes**: Drag & drop PDF files or load from cloud
3. **Start Analysis**: Click "Start Analysis" and wait for results
4. **Review Results**: Browse candidates in different categories
5. **Send Emails**: Bulk notify candidates about their status
6. **Export Data**: Download reports and candidate data

## 📊 Sample Results

```
📄 Processing Results:
✅ Shortlisted: 15 candidates (Score ≥ 75%)
🟨 Under Review: 25 candidates (Score 40-74%)
❌ Rejected: 60 candidates (Score < 40%)

⏱️ Total Processing Time: 4.2 minutes for 100 resumes
```

## 🔧 Main Files Explained

- **app.py** - Main web interface (what users see)
- **backend.py** - AI analysis engine (the smart brain)  
- **utils.py** - Helper functions (PDF reading, contact extraction)
- **pdf_utils.py** - Report generation (creates PDF summaries)
- **gmail_to_azure.py** - Email automation (fetches resumes from Gmail)
- **constants.py** - Configuration settings
- **.env** - Secret keys and passwords
