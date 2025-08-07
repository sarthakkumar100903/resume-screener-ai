from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import asyncio
import os
from datetime import datetime
from azure.storage.blob import BlobServiceClient

from constants import AZURE_CONFIG
from utils import (
    parse_resume,
    get_text_chunks,
    get_embedding_cached,
    get_cosine_similarity,
    extract_contact_info
)
from backend import get_resume_analysis_async, extract_role_from_jd
# from email_generator import schedule_interview

# Optional tabbed UI
try:
    from display_section import show_candidate_tabs
    USE_TABS = True
except ImportError:
    USE_TABS = False

# Initialize session state
if "candidate_df" not in st.session_state:
    st.session_state["candidate_df"] = None
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

st.set_page_config(layout="wide", page_title="AI Resume Screener")
st.markdown("<h1 style='text-align:center;'>🤖 AI Resume Screener</h1>", unsafe_allow_html=True)
st.markdown("---")

# ====== Sidebar ======
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

    analyze = st.button("🚀 Analyze from Azure Storage")

# ====== Azure Blob Loader ======
def load_resumes_from_blob():
    connection_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = AZURE_CONFIG["resumes_container"]
    blob_service_client = BlobServiceClient.from_connection_string(connection_str)
    container_client = blob_service_client.get_container_client(container)
    resumes = []
    for blob in container_client.list_blobs():
        if blob.name.lower().endswith(".pdf"):
            blob_client = container_client.get_blob_client(blob.name)
            content = blob_client.download_blob().readall()
            resumes.append((blob.name, content))
    return resumes

# ====== Verdict Logic ======
def verdict_logic(row):
    if not row["email"]:
        return "Missing Info"
    if (
        row["jd_score"] >= jd_thresh
        and row["skills_score"] >= skill_thresh
        and row["domain_score"] >= domain_thresh
        and row["experience_score"] >= exp_thresh
        and row["score"] >= score_thresh
    ):
        return "Shortlisted"
    else:
        return "Rejected"

# ====== Main Processing ======
if jd and analyze and not st.session_state["analysis_done"]:
    resumes_from_blob = load_resumes_from_blob()
    total = len(resumes_from_blob)

    if total == 0:
        st.warning("No valid PDF resumes found in Azure Blob.")
    else:
        progress = st.progress(0, text="Analyzing resumes...")
        jd_embedding = get_embedding_cached(jd)
        results = []

        async def process_all():
            tasks = []
            for idx, (file_name, file_bytes) in enumerate(resumes_from_blob):
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
        for i in range(len(resumes_from_blob)):
            progress.progress(i / total, text=f"Processing {i+1} of {total}...")
        results = loop.run_until_complete(process_all())
        loop.close()

        # Final step: DataFrame, sorting, verdict, filtering, email
        df = pd.DataFrame(results)

        # Sort by score DESCENDING
        df = df.sort_values("score", ascending=False).copy()

        # Apply Top-N filter
        if top_n > 0:
            df = df.iloc[:top_n].copy()

        # Assign verdicts
        df["verdict"] = df.apply(verdict_logic, axis=1)

        # Schedule/send emails based on verdict
        for _, row in df.iterrows():
            schedule_interview(
                name=row.get("name"),
                email=row.get("email"),
                verdict=row.get("verdict"),
                role=role,
                jd=jd
            )

        st.session_state["candidate_df"] = df
        st.session_state["analysis_done"] = True
        st.success("✅ Resume analysis complete!")

# ========== Display ==========
if st.session_state["candidate_df"] is not None:
    if USE_TABS:
        show_candidate_tabs(
            st.session_state["candidate_df"],
            score_thresh,
            jd_thresh,
            skill_thresh,
            domain_thresh,
            exp_thresh
        )
    else:
        st.dataframe(st.session_state["candidate_df"])
