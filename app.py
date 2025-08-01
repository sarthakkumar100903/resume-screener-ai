# ✅ Combined app.py: Merged features from app.py and app (2).py
# This file includes full Streamlit UI, resume analysis, emailing, scheduling, analytics

from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import base64
import asyncio
from datetime import datetime, timedelta
import uuid

# Optional (used in app (2).py)
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

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
from email_generator import send_email, check_missing_info, send_missing_info_email, schedule_interview

SCOPES = ['https://www.googleapis.com/auth/calendar.events']

# Session state init
if "candidate_df" not in st.session_state:
    st.session_state["candidate_df"] = None
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

st.set_page_config(layout="wide", page_title="AI Resume Screener")
st.markdown("<h1 style='text-align:center;'>🤖 AI Resume Screener</h1>", unsafe_allow_html=True)
st.markdown("---")

# ========== Sidebar Inputs ==========
with st.sidebar:
    jd = st.text_area("📄 Paste Job Description", height=200)
    role = extract_role_from_jd(jd) if jd else "N/A"
    if jd:
        st.markdown(f"🧠 **Extracted Role:** `{role}`")

    domain = st.text_input("🏢 Preferred Domain", "")
    skills = st.text_area("🛠️ Required Skills (comma separated)", "")
    exp_range = st.selectbox("📈 Required Experience", ["0–1 yrs", "1–3 yrs", "2–4 yrs", "4+ yrs"])

    st.markdown("### 🎚️ Thresholds")
    jd_thresh = st.slider("JD Similarity", 0, 100, 50)
    skill_thresh = st.slider("Skills Match", 0, 100, 50)
    domain_thresh = st.slider("Domain Match", 0, 100, 50)
    exp_thresh = st.slider("Experience Match", 0, 100, 50)
    score_thresh = st.slider("Final Score Threshold", 0, 100, 50)
    top_n = st.number_input("🎯 Top-N Candidates", 0, value=0)

    uploaded_files = st.file_uploader("📤 Upload Resumes (PDF)", type=["pdf"], accept_multiple_files=True)
    analyze = st.button("🚀 Analyze")

# ========== Processing ==========
if jd and uploaded_files and analyze and not st.session_state["analysis_done"]:
    progress = st.progress(0, text="Starting Analysis...")
    total = len(uploaded_files)
    jd_embedding = get_embedding_cached(jd)
    results = []

    async def process_all():
        tasks = []
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
                jd=jd,
                resume_text=resume_text,
                contact=contact,
                role=role,
                domain=domain,
                skills=skills,
                experience_range=exp_range,
                jd_similarity=jd_sim,
                resume_file=file_name
            )
            tasks.append(task)
        return await asyncio.gather(*tasks)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    for i in range(len(uploaded_files)):
        progress.progress(i / total, text=f"Processing {i+1} of {total}...")
    results = loop.run_until_complete(process_all())
    loop.close()

    for r in results:
        r["recruiter_notes"] = ""
        if r["score"] < score_thresh and r["verdict"] != "reject":
            r["verdict"] = "reject"
            r.setdefault("reasons_if_rejected", []).append(f"Score below threshold {r['score']} < {score_thresh}")

    st.success("✅ All resumes processed!")
    df = pd.DataFrame(results).fillna("N/A")

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

    if top_n > 0:
        sorted_df = df.sort_values("score", ascending=False)
        top = sorted_df.head(top_n).copy()
        top["verdict"] = "shortlist"
        rest = sorted_df.iloc[top_n:].copy()
        rest["verdict"] = rest["verdict"].apply(lambda v: v if v == "reject" else "review")
        df = pd.concat([top, rest], ignore_index=True)

    st.session_state["candidate_df"] = df
    st.session_state["analysis_done"] = True

# ========== Display & Actions ==========
if st.session_state["candidate_df"] is not None:
    from display_section import show_candidate_tabs  # modularize tab UI
    show_candidate_tabs(st.session_state["candidate_df"], score_thresh, jd_thresh, skill_thresh, domain_thresh, exp_thresh)
