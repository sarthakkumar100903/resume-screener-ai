from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import asyncio
import os
from azure.storage.blob import BlobServiceClient

from constants import AZURE_CONFIG
from utils import (
    parse_resume,
    get_text_chunks,
    get_embedding_cached,
    get_cosine_similarity,
    extract_contact_info,
    upload_to_blob
)
from backend import get_resume_analysis_async, extract_role_from_jd
from email_generator import schedule_interview

# Google OAuth (optional)
SCOPES = ['https://www.googleapis.com/auth/calendar.events']

try:
    from display_section import show_candidate_tabs
    USE_TABS = True
except ImportError:
    USE_TABS = False

if "candidate_df" not in st.session_state:
    st.session_state["candidate_df"] = None
if "analysis_done" not in st.session_state:
    st.session_state["analysis_done"] = False

st.set_page_config(layout="wide", page_title="AI Resume Screener")
st.markdown("<h1 style='text-align:center;'>🤖 AI Resume Screener</h1>", unsafe_allow_html=True)
st.markdown("---")

# ========== Sidebar ==========
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

# ========== Azure Blob Loader ==========
def load_resumes_from_blob():
    connection_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = AZURE_CONFIG["resumes_container"]
    blob_service_client = BlobServiceClient.from_connection_string(connection_str)
    container_client = blob_service_client.get_container_client(container)
    blobs = container_client.list_blobs()

    resumes = []
    for blob in blobs:
        if blob.name.lower().endswith(".pdf"):
            blob_client = container_client.get_blob_client(blob.name)
            content = blob_client.download_blob().readall()
            resumes.append((blob.name, content))
    return resumes

# ========== Resume Analysis ==========
if jd and analyze and not st.session_state["analysis_done"]:
    resumes_from_blob = load_resumes_from_blob()
    total = len(resumes_from_blob)

    if total == 0:
        st.warning("No valid PDF resumes found in Azure Blob.")
    else:
        progress = st.progress(0, text="Analyzing resumes...")
        jd_embedding = get_embedding_cached(jd)

        async def process_all():
            tasks = []
            for file_name, file_bytes in resumes_from_blob:
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
        results = loop.run_until_complete(process_all())
        loop.close()

        df = pd.DataFrame(results).fillna("N/A")

        # Apply thresholds and update verdicts
        def verdict_logic(row):
            if row["verdict"] == "reject":
                return "reject"
            if (
                row.get("jd_similarity", 0) < jd_thresh or
                row.get("skills_match", 0) < skill_thresh or
                row.get("domain_match", 0) < domain_thresh or
                row.get("experience_match", 0) < exp_thresh
            ):
                return "review"
            return "shortlist"

        df["verdict"] = df.apply(verdict_logic, axis=1)

        # Apply score threshold
        for i, r in df.iterrows():
            if r.get("score", 0) < score_thresh and df.at[i, "verdict"] != "reject":
                df.at[i, "verdict"] = "reject"
                reasons = r.get("reasons_if_rejected", [])
                if isinstance(reasons, str):
                    reasons = [reasons]
                reasons.append(f"Score below threshold {r['score']} < {score_thresh}")
                df.at[i, "reasons_if_rejected"] = reasons

        # Top-N filter
        if top_n > 0:
            df = df.sort_values("score", ascending=False)
            top_df = df.head(top_n).copy()
            top_df["verdict"] = "shortlist"
            rest = df.iloc[top_n:].copy()
            rest["verdict"] = rest["verdict"].apply(lambda v: v if v == "reject" else "review")
            df = pd.concat([top_df, rest], ignore_index=True)

        st.success("✅ Resumes processed from Azure!")
        st.session_state["candidate_df"] = df
        st.session_state["analysis_done"] = True

# ========== Display Results ==========
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
