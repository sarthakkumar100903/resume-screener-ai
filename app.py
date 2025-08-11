import streamlit as st
import pandas as pd
import base64
import asyncio
from datetime import datetime
from io import BytesIO
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import uuid
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import datetime
from datetime import timedelta

SCOPES = ['https://www.googleapis.com/auth/calendar.events']
from constants import AZURE_CONFIG
from utils import (
    parse_resume,
    get_text_chunks,
    get_embedding_cached,
    get_cosine_similarity,
    upload_to_blob,
    extract_contact_info,
    save_summary_to_blob,
    save_csv_to_blob
)
from backend import get_resume_analysis_async, extract_role_from_jd
from pdf_utils import generate_summary_pdf
from email_generator import send_email, check_missing_info, send_missing_info_email

# === NEW IMPORTS for Azure Blob ===
from azure.storage.blob import BlobServiceClient

# =========Enhanced Design==========
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Main app styling */
    .stApp {
        background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #0f1419 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e2a3a 0%, #0f1419 100%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Header styling */
    header[data-testid="stHeader"] {
        background: transparent !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Custom title styling */
    .main-title {
        text-align: center;
        background: linear-gradient(135deg, #00d4ff 0%, #5865f2 50%, #ff6b6b 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 0 30px rgba(0, 212, 255, 0.3);
    }

    .subtitle {
        text-align: center;
        color: #94a3b8;
        font-size: 1.4rem;
        font-weight: 400;
        margin-bottom: 2rem;
    }

    /* Card styling for candidates */
    .candidate-card {
        background: linear-gradient(135deg, rgba(30, 42, 58, 0.8) 0%, rgba(15, 20, 25, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
    }

    .candidate-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 212, 255, 0.2);
        border-color: rgba(0, 212, 255, 0.3);
    }

    /* Text styling */
    html, body, [class*="css"], p, div, span, label, h1, h2, h3, h4, h5, h6 {
        color: #f1f5f9 !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Input field styling */
    textarea, input, select {
        background: rgba(30, 42, 58, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        color: #f1f5f9 !important;
        backdrop-filter: blur(10px) !important;
    }

    textarea:focus, input:focus, select:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 2px rgba(0, 212, 255, 0.2) !important;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #5865f2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.5rem !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4) !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(30, 42, 58, 0.5);
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        background: transparent;
        border: 1px solid transparent;
        transition: all 0.3s ease;
    }

    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 212, 255, 0.1);
        border-color: rgba(0, 212, 255, 0.3);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00d4ff 0%, #5865f2 100%) !important;
        border-color: transparent !important;
    }

    /* Metric styling */
    .metric-container {
        background: rgba(30, 42, 58, 0.6);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Status badges */
    .status-shortlist {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .status-review {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .status-reject {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
    }

    /* Progress bar styling */
    .stProgress .st-bo {
        background: linear-gradient(90deg, #00d4ff 0%, #5865f2 100%) !important;
    }

    /* Sidebar section headers */
    .sidebar-section {
        background: rgba(0, 212, 255, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 3px solid #00d4ff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# === Initialize BlobServiceClient once ===
blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONFIG["connection_string"])
resumes_container_client = blob_service_client.get_container_client(AZURE_CONFIG["resumes_container"])

# === Function to download all PDF blobs from container ===
def download_all_pdf_blobs():
    blobs = resumes_container_client.list_blobs()
    pdf_files = []
    for blob in blobs:
        if blob.name.lower().endswith(".pdf"):
            downloader = resumes_container_client.download_blob(blob.name)
            pdf_bytes = downloader.readall()
            pdf_files.append((blob.name, pdf_bytes))
    return pdf_files

# =========================
# Session state setup & page config
if "candidate_df" not in st.session_state:
    st.session_state["candidate_df"] = None
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

st.set_page_config(
    layout="wide", 
    page_title="EazyAI Resume Screener",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# Enhanced Header
st.markdown('<h1 class="main-title">EazyAI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Intelligent Resume Screening Platform</p>', unsafe_allow_html=True)
st.markdown("---")

# ========== Sidebar Inputs ==========
with st.sidebar:
    st.markdown('<div class="sidebar-section"><h3>📋 Job Configuration</h3></div>', unsafe_allow_html=True)
    
    jd = st.text_area("📄 Paste Job Description", height=200, placeholder="Enter the complete job description here...")
    
    role = "N/A"
    if jd:
        role = extract_role_from_jd(jd)
        if role != "N/A":
            st.success(f"🎯 **Detected Role:** {role}")
        else:
            st.warning("⚠️ Could not extract role from JD")

    domain = st.text_input("🏢 Preferred Domain", placeholder="e.g., Healthcare, Fintech, E-commerce")
    skills = st.text_area("🛠️ Required Skills (comma separated)", placeholder="Python, React, AWS, Machine Learning")
    exp_range = st.selectbox("📈 Required Experience", ["0–1 yrs", "1–3 yrs", "2–4 yrs", "4+ yrs"])

    st.markdown('<div class="sidebar-section"><h3>🎚️ Matching Thresholds</h3></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        jd_thresh = st.slider("JD Similarity", 0, 100, 50, help="Minimum similarity with job description")
        domain_thresh = st.slider("Domain Match", 0, 100, 50, help="Minimum domain experience match")
    with col2:
        skill_thresh = st.slider("Skills Match", 0, 100, 50, help="Minimum required skills match")
        exp_thresh = st.slider("Experience Match", 0, 100, 50, help="Experience level compatibility")
    
    score_thresh = st.slider("🎯 Final Score Threshold", 0, 100, 50, help="Overall qualification threshold")
    top_n = st.number_input("🏆 Top-N Candidates", 0, 50, 0, help="Limit shortlisted candidates (0 = no limit)")

    st.markdown('<div class="sidebar-section"><h3>📂 Resume Source</h3></div>', unsafe_allow_html=True)
    
    load_from_blob = st.checkbox("☁️ Load from Azure Blob Storage", value=False)

    if not load_from_blob:
        uploaded_files = st.file_uploader(
            "📤 Upload Resume Files", 
            type=["pdf"], 
            accept_multiple_files=True,
            help="Select multiple PDF resume files"
        )
    else:
        uploaded_files = None

    st.markdown("---")
    analyze = st.button("🚀 Start Analysis", type="primary", use_container_width=True)

# ========== Processing ==========
if jd and analyze and not st.session_state["analysis_done"]:
    progress_container = st.container()
    with progress_container:
        st.markdown("### 🔄 Processing Resumes...")
        progress = st.progress(0, text="Initializing analysis...")

    # Load resumes
    if load_from_blob:
        blob_files = download_all_pdf_blobs()
        total = len(blob_files)
        if total == 0:
            st.error("❌ No PDF resumes found in Azure Blob storage container.")
            st.stop()
        else:
            st.info(f"📊 Found {total} resumes in blob storage")
    else:
        blob_files = None
        total = len(uploaded_files) if uploaded_files else 0
        if total == 0:
            st.error("❌ Please upload at least one resume or enable blob storage option.")
            st.stop()

    results = []
    jd_embedding = get_embedding_cached(jd)

    async def process_all():
        tasks = []

        if load_from_blob:
            for idx, (file_name, file_bytes) in enumerate(blob_files):
                upload_to_blob(file_bytes, file_name, AZURE_CONFIG["resumes_container"])
                resume_text = parse_resume(file_bytes)
                contact = extract_contact_info(resume_text)

                chunks = get_text_chunks(resume_text)
                resume_embedding = get_embedding_cached(" ".join(chunks))
                jd_sim = round(get_cosine_similarity(resume_embedding, jd_embedding) * 100, 2)

                task = get_resume_analysis_async(
                    jd=jd, resume_text=resume_text, contact=contact, role=role,
                    domain=domain, skills=skills, experience_range=exp_range,
                    jd_similarity=jd_sim, resume_file=file_name
                )
                tasks.append(task)
        else:
            for idx, file in enumerate(uploaded_files):
                file_bytes = file.read()
                file_name = file.name.replace(".pdf", "")
                upload_to_blob(file_bytes, file_name + ".pdf", AZURE_CONFIG["resumes_container"])

                resume_text = parse_resume(file_bytes)
                contact = extract_contact_info(resume_text)

                chunks = get_text_chunks(resume_text)
                resume_embedding = get_embedding_cached(" ".join(chunks))
                jd_sim = round(get_cosine_similarity(resume_embedding, jd_embedding) * 100, 2)

                task = get_resume_analysis_async(
                    jd=jd, resume_text=resume_text, contact=contact, role=role,
                    domain=domain, skills=skills, experience_range=exp_range,
                    jd_similarity=jd_sim, resume_file=file_name
                )
                tasks.append(task)

        return await asyncio.gather(*tasks)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Progress updates
    for i in range(total):
        progress.progress((i + 1) / total, text=f"Analyzing resume {i+1} of {total}...")

    results = loop.run_until_complete(process_all())
    loop.close()

    # Process results
    for r in results:
        r["recruiter_notes"] = ""
        if r["score"] < score_thresh and r["verdict"] != "reject":
            r["verdict"] = "reject"
            r.setdefault("reasons_if_rejected", []).append(
                f"Score below threshold {r['score']} < {score_thresh}"
            )

    progress.progress(100, text="✅ Analysis completed!")
    st.success(f"🎉 Successfully processed {len(results)} resumes!")
    
    df = pd.DataFrame(results).fillna("N/A")
    df.replace("n/a", "N/A", regex=True, inplace=True)

    # Enhanced verdict logic
    def verdict_logic(row):
        if row["verdict"] == "reject":
            return "reject"
        elif (
            row["jd_similarity"] < jd_thresh or
            row["skills_match"] < skill_thresh or
            row["domain_match"] < domain_thresh or
            row["experience_match"] < exp_thresh
        ):
            return "review"
        return "shortlist"

    df["verdict"] = df.apply(verdict_logic, axis=1)

    # Top-N logic
    if top_n > 0:
        sorted_df = df.sort_values("score", ascending=False)
        top = sorted_df.head(top_n).copy()
        top["verdict"] = "shortlist"
        rest = sorted_df.iloc[top_n:].copy()
        rest["verdict"] = rest["verdict"].apply(lambda v: v if v == "reject" else "review")
        df = pd.concat([top, rest], ignore_index=True)

    st.session_state["candidate_df"] = df
    st.session_state["analysis_done"] = True

# ========== Display Results ==========
if st.session_state["candidate_df"] is not None:
    df = st.session_state["candidate_df"]
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    shortlisted_count = len(df[df["verdict"] == "shortlist"])
    review_count = len(df[df["verdict"] == "review"]) 
    rejected_count = len(df[df["verdict"] == "reject"])
    total_count = len(df)
    
    with col1:
        st.markdown(f'<div class="metric-container"><h3>✅ {shortlisted_count}</h3><p>Shortlisted</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-container"><h3>🟨 {review_count}</h3><p>Under Review</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-container"><h3>❌ {rejected_count}</h3><p>Rejected</p></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-container"><h3>📊 {total_count}</h3><p>Total Processed</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    
    # Enhanced tabs
    tabs = st.tabs([
        f"✅ Shortlisted ({shortlisted_count})", 
        f"🟨 Under Review ({review_count})", 
        f"❌ Rejected ({rejected_count})", 
        "📊 Analytics Dashboard"
    ])

    # Process each verdict tab
    for verdict, tab in zip(["shortlist", "review", "reject"], tabs[:3]):
        with tab:
            filtered = df[df["verdict"] == verdict]
            
            if len(filtered) == 0:
                st.info(f"No candidates in {verdict} category")
                continue
                
            # Bulk actions for rejected candidates
            if verdict == "reject" and len(filtered) > 0:
                st.markdown("### 📧 Bulk Actions")
                if st.button(f"📬 Send Bulk Rejection Emails ({len(filtered)} candidates)", 
                           key=f"bulk_reject_{verdict}", type="secondary"):
                    sent_count = 0
                    for _, row in filtered.iterrows():
                        email = row.get('email', '')
                        if email and email != 'N/A':
                            subject = "Application Status Update"
                            body = f"""Dear {row['name']},

Thank you for your interest in our {role} position. After careful consideration, we have decided not to proceed with your application at this time.

We appreciate the time you invested in the application process and wish you success in your future endeavors.

Best regards,
Recruitment Team"""
                            
                            if send_email(email, subject, body):
                                sent_count += 1
                    
                    if sent_count > 0:
                        st.success(f"✅ Successfully sent {sent_count} rejection emails")
                    else:
                        st.warning("⚠️ No emails were sent")
                
                st.markdown("---")

            # Display candidates
            for idx, (i, row) in enumerate(filtered.iterrows()):
                with st.container():
                    st.markdown('<div class="candidate-card">', unsafe_allow_html=True)
                    
                    # Header section
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Candidate name with status badge
                        status_class = f"status-{verdict}"
                        st.markdown(f"""
                        <h3>👤 {row['name']} <span class="{status_class}">{verdict.upper()}</span></h3>
                        """, unsafe_allow_html=True)
                        
                        # Contact info
                        st.markdown(f"📧 **{row.get('email', 'N/A')}** | 📞 **{row.get('phone', 'N/A')}**")
                        
                        # Fitment summary
                        st.markdown(f"💡 **Fitment:** {row.get('fitment', 'N/A')}")
                        
                        # Score metrics with visual indicators
                        col_jd, col_skills, col_domain, col_exp, col_final = st.columns(5)
                        with col_jd:
                            st.metric("JD Match", f"{row['jd_similarity']}%")
                        with col_skills:
                            st.metric("Skills", f"{row['skills_match']}%")
                        with col_domain:
                            st.metric("Domain", f"{row['domain_match']}%")
                        with col_exp:
                            st.metric("Experience", f"{row['experience_match']}%")
                        with col_final:
                            st.metric("Final Score", f"{row['score']}%")

                        # Interactive elements with unique keys
                        unique_suffix = f"{verdict}_{i}_{idx}_{uuid.uuid4().hex[:8]}"
                        
                        note_key = f"note_{unique_suffix}"
                        verdict_key = f"verdict_{unique_suffix}"
                        
                        # Initialize session state
                        if note_key not in st.session_state:
                            st.session_state[note_key] = row.get("recruiter_notes", "")
                        if verdict_key not in st.session_state:
                            st.session_state[verdict_key] = row["verdict"]

                        # Input widgets
                        new_note = st.text_area(
                            "📝 Recruiter Notes", 
                            value=st.session_state[note_key], 
                            key=note_key,
                            height=100
                        )
                        
                        new_verdict = st.selectbox(
                            "🔄 Update Status", 
                            ["shortlist", "review", "reject"],
                            index=["shortlist", "review", "reject"].index(st.session_state[verdict_key]),
                            key=verdict_key
                        )

                        # Update dataframe
                        df.at[i, "recruiter_notes"] = new_note
                        df.at[i, "verdict"] = new_verdict

                    with col2:
                        # Action buttons
                        st.markdown("### 🎬 Actions")
                        
                        # Email button with unique key
                        email_key = f"email_{unique_suffix}"
                        if st.button(f"✉️ Send Email", key=email_key, type="primary"):
                            email = row.get('email', '')
                            if email and email != 'N/A':
                                if verdict == "shortlist":
                                    subject = "Congratulations! You've been shortlisted"
                                    body = f"""Dear {row['name']},

Congratulations! After reviewing your application for the {role} position, we are pleased to inform you that you have been shortlisted for the next round.

Our team was impressed with your qualifications and experience. We will be in touch soon with details about the next steps in our selection process.

Best regards,
Recruitment Team"""
                                
                                elif verdict == "review":
                                    subject = "Additional Information Required"
                                    body = f"""Dear {row['name']},

Thank you for your application for the {role} position. 

We are currently reviewing your profile and may need some additional information to proceed. Our recruitment team will be in touch shortly with any questions.

Thank you for your patience.

Best regards,
Recruitment Team"""
                                
                                else:  # reject
                                    subject = "Application Status Update"
                                    body = f"""Dear {row['name']},

Thank you for your interest in our {role} position. After careful consideration, we have decided not to proceed with your application at this time.

We appreciate your time and interest in our organization.

Best regards,
Recruitment Team"""
                                
                                if send_email(email, subject, body):
                                    st.success("✅ Email sent!")
                                else:
                                    st.error("❌ Email failed")
                            else:
                                st.error("❌ No valid email")

                        # Download summary
                        if st.button(f"📄 Generate Summary", key=f"summary_{unique_suffix}"):
                            try:
                                pdf_bytes = generate_summary_pdf(row)
                                summary_name = f"{row['name'].replace(' ', '_')}_Summary.pdf"
                                save_summary_to_blob(pdf_bytes, summary_name, AZURE_CONFIG["summaries_container"])
                                
                                b64 = base64.b64encode(pdf_bytes).decode()
                                st.markdown(f'''
                                <a href="data:application/octet-stream;base64,{b64}" 
                                   download="{summary_name}" 
                                   style="color: #00d4ff; text-decoration: none;">
                                   📥 Download Summary PDF
                                </a>
                                ''', unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"❌ Summary generation failed: {str(e)}")

                    st.markdown('</div>', unsafe_allow_html=True)
                    st.markdown("---")

            # Export functionality
            if len(filtered) > 0:
                st.markdown("### 📤 Export Data")
                export_df = filtered.drop(columns=["resume_text", "embedding"], errors="ignore")
                csv_name = f"{verdict}_candidates_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        "📊 Download CSV", 
                        export_df.to_csv(index=False),
                        file_name=csv_name,
                        mime="text/csv",
                        key=f"csv_download_{verdict}"
                    )
                with col2:
                    if st.button(f"☁️ Save to Blob", key=f"blob_save_{verdict}"):
                        save_csv_to_blob(export_df, csv_name, AZURE_CONFIG["csv_container"])
                        st.success("✅ Saved to Azure Blob!")

    # ========== Analytics Tab ==========
    with tabs[3]:
        st.markdown("### 📊 Comprehensive Analytics")
        
        # Summary statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Verdict Distribution")
            verdict_counts = df["verdict"].value_counts()
            st.bar_chart(verdict_counts)
            
            # Score statistics
            st.markdown("#### 🎯 Score Statistics")
            st.write(f"**Average Score:** {df['score'].mean():.1f}%")
            st.write(f"**Highest Score:** {df['score'].max():.1f}%")
            st.write(f"**Lowest Score:** {df['score'].min():.1f}%")
        
        with col2:
            st.markdown("#### 📊 Score Distribution")
            score_cols = ["jd_similarity", "skills_match", "domain_match", "experience_match", "score"]
            st.line_chart(df[score_cols])
            
            # Top performers
            st.markdown("#### 🏆 Top 5 Candidates")
            top_candidates = df.nlargest(5, 'score')[['name', 'score', 'verdict']]
            st.dataframe(top_candidates, hide_index=True)

        # Detailed data table
        st.markdown("#### 🗂️ Complete Dataset")
        display_df = df.drop(columns=["resume_text", "embedding"], errors="ignore")
        st.dataframe(display_df, use_container_width=True)

        # Fraud detection
        flagged = df[df["fraud_detected"] == True]
        if not flagged.empty:
            st.markdown("#### 🚨 Flagged Profiles")
            st.error(f"Found {len(flagged)} profiles with potential issues:")
            flagged_display = flagged[["name", "red_flags", "missing_gaps", "email"]]
            st.dataframe(flagged_display, use_container_width=True)
        else:
            st.success("✅ No fraud or red flags detected in any profiles")

elif not st.session_state["analysis_done"]:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 3rem 2rem; background: rgba(30, 42, 58, 0.6); border-radius: 16px; margin: 2rem 0;">
        <h2>🚀 Welcome to EazyAI Resume Screener</h2>
        <p style="font-size: 1.2rem; color: #94a3b8; margin-bottom: 2rem;">
            Streamline your hiring process with AI-powered resume analysis
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem; margin-top: 2rem;">
            <div style="background: rgba(0, 212, 255, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(0, 212, 255, 0.2);">
                <h4>📋 Step 1</h4>
                <p>Paste your job description in the sidebar</p>
            </div>
            <div style="background: rgba(88, 101, 242, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(88, 101, 242, 0.2);">
                <h4>⚙️ Step 2</h4>
                <p>Configure matching thresholds and criteria</p>
            </div>
            <div style="background: rgba(16, 185, 129, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(16, 185, 129, 0.2);">
                <h4>📤 Step 3</h4>
                <p>Upload resumes or load from Azure Blob</p>
            </div>
            <div style="background: rgba(255, 107, 107, 0.1); padding: 1.5rem; border-radius: 12px; border: 1px solid rgba(255, 107, 107, 0.2);">
                <h4>🚀 Step 4</h4>
                <p>Click 'Start Analysis' and review results</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🧠 AI-Powered Analysis
        - Intelligent role extraction from JD
        - Semantic similarity matching
        - Skills and experience evaluation
        - Automated fraud detection
        """)
    
    with col2:
        st.markdown("""
        ### ⚡ Advanced Features  
        - Bulk email automation
        - PDF summary generation
        - Azure Blob integration
        - Real-time progress tracking
        """)
    
    with col3:
        st.markdown("""
        ### 📊 Smart Insights
        - Comprehensive analytics dashboard
        - Customizable scoring thresholds
        - Interactive candidate management
        - Export capabilities
        """)
    
    st.markdown("---")
    st.info("👈 **Get Started:** Fill in the job description and configuration options in the sidebar, then click 'Start Analysis' to begin!")
