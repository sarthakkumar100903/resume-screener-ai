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
    extract_contact_info,
    upload_to_blob
)
from backend import get_resume_analysis_async, extract_role_from_jd

# Google OAuth (optional)
SCOPES = ['https://www.googleapis.com/auth/calendar.events']

# Try importing display_section
try:
    from display_section import show_candidate_tabs
    USE_TABS = True
except ImportError:
    USE_TABS = False

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

    analyze = st.button("🚀 Analyze from Azure Storage")

# ========== Processing ==========

def load_resumes_from_blob():
    connection_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    container = AZURE_CONFIG["resumes_container"]
    blob_service_client = BlobServiceClient.from_connection_string(connection_str)
    container_client = blob_service_client.get_container_client(container)
    blobs = container_client.list_blobs()

    resumes = []
    for blob in blobs:
        if not blob.name.lower().endswith(".pdf"):
            continue
        blob_client = container_client.get_blob_client(blob.name)
        content = blob_client.download_blob().readall()
        resumes.append((blob.name, content))
    return resumes

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

        for r in results:
            r["recruiter_notes"] = ""
            if r["score"] < score_thresh and r["verdict"] != "reject":
                r["verdict"] = "reject"
                r.setdefault("reasons_if_rejected", []).append(f"Score below threshold {r['score']} < {score_thresh}")

        st.success("✅ Resumes processed from Azure!")
        df = pd.DataFrame(results).fillna("N/A")
        
        # Step 1: Apply verdicts
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
        
        # Step 2: Always sort by score descending
        df = df.sort_values("score", ascending=False).copy()
        
        # Step 3: Apply Top-N after sorting
        if top_n > 0:
            df = df.head(top_n).copy()
            df["verdict"] = "shortlist"  # Optional: force verdict if needed
        
        # Step 4: Store to session state
        st.session_state["candidate_df"] = df
        st.session_state["analysis_done"] = True


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

# ===== Email Action Buttons =====
if st.session_state["candidate_df"] is not None:
    st.markdown("### 📧 Send Emails to Candidates")
    df = st.session_state["candidate_df"]

    from email_generator import (
        send_selection_email,
        send_rejection_email,
        send_missing_info_email,
        check_missing_info
    )

    table_html = """
    <table style="width:100%; border-collapse: collapse;">
        <thead>
            <tr style="background-color:#f2f2f2; text-align:left;">
                <th style="padding:8px; border:1px solid #ddd;">Name</th>
                <th style="padding:8px; border:1px solid #ddd;">Email</th>
                <th style="padding:8px; border:1px solid #ddd;">Verdict</th>
                <th style="padding:8px; border:1px solid #ddd;">Score</th>
                <th style="padding:8px; border:1px solid #ddd;">Actions</th>
            </tr>
        </thead>
        <tbody>
    """

    for idx, row in df.iterrows():
        col1, col2, col3 = st.columns(3)
        with col1:
            select_btn = st.button(f"✅", key=f"select_{idx}")
        with col2:
            reject_btn = st.button(f"❌", key=f"reject_{idx}")
        with col3:
            missing_btn = st.button(f"📩", key=f"missing_{idx}")

        if select_btn:
            send_selection_email(row["email"], row["name"])
            st.success(f"Selection email sent to {row['name']}")
        if reject_btn:
            send_rejection_email(row["email"], row["name"])
            st.success(f"Rejection email sent to {row['name']}")
        if missing_btn:
            missing_fields = check_missing_info(row)
            if missing_fields:
                send_missing_info_email(row["email"], row["name"], missing_fields)
                st.success(f"Missing info email sent to {row['name']}")
            else:
                st.info(f"No missing fields for {row['name']}")

        table_html += f"""
            <tr>
                <td style='padding:8px; border:1px solid #ddd;'>{row['name']}</td>
                <td style='padding:8px; border:1px solid #ddd;'>{row['email']}</td>
                <td style='padding:8px; border:1px solid #ddd;'>{row['verdict']}</td>
                <td style='padding:8px; border:1px solid #ddd;'>{row['score']}</td>
                <td style='padding:8px; border:1px solid #ddd;'>
                    ✅ Select | ❌ Reject | 📩 Missing Info
                </td>
            </tr>
        """

    table_html += "</tbody></table>"
    st.markdown(table_html, unsafe_allow_html=True)
