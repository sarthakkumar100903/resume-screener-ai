import streamlit as st
from backend import analyze_resume
from email_generator import (
    send_selection_email,
    send_rejection_email,
    send_missing_info_email,
    check_missing_info,
)
from azure.storage.blob import BlobServiceClient
import pandas as pd
import io
import os

# Azure Blob config
AZURE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER_NAME = "resumes"

st.title("AI Resume Screener")

# Sidebar inputs
job_title = st.sidebar.text_input("Job Title")
job_description = st.sidebar.text_area("Job Description")
score_thresh = st.sidebar.slider("Total Score Threshold", 0, 100, 60)
jd_thresh = st.sidebar.slider("JD Match Threshold", 0, 100, 50)
skill_thresh = st.sidebar.slider("Skill Match Threshold", 0, 100, 50)
domain_thresh = st.sidebar.slider("Domain Match Threshold", 0, 100, 50)
exp_thresh = st.sidebar.slider("Experience Match Threshold", 0, 100, 50)
top_n = st.sidebar.number_input("Top N Candidates", min_value=1, value=5)

if st.sidebar.button("Analyze Resumes"):
    if not job_title or not job_description:
        st.warning("Please enter both Job Title and Job Description.")
    else:
        st.info("Fetching and analyzing resumes...")

        # Connect to Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)

        candidate_data = []

        for blob in container_client.list_blobs():
            blob_client = container_client.get_blob_client(blob)
            content = blob_client.download_blob().readall()

            try:
                file_bytes = io.BytesIO(content)
                result = analyze_resume(file_bytes, job_description, job_title)

                if result:
                    result["filename"] = blob.name
                    candidate_data.append(result)
            except Exception as e:
                st.error(f"Error analyzing {blob.name}: {e}")

        if not candidate_data:
            st.warning("No valid resumes found.")
        else:
            df = pd.DataFrame(candidate_data)
            df["verdict"] = df.apply(lambda row: "Rejected", axis=1)  # default

            for idx, row in df.iterrows():
                missing = check_missing_info(row)

                if missing:
                    df.at[idx, "verdict"] = "Missing Info"
                    send_missing_info_email(row["email"], row.get("name", ""), missing)

                elif (
                    row["score"] >= score_thresh and
                    row["jd_similarity"] >= jd_thresh and
                    row["skill_match_score"] >= skill_thresh and
                    row["domain_match_score"] >= domain_thresh and
                    row["experience_match_score"] >= exp_thresh
                ):
                    df.at[idx, "verdict"] = "Selected"
                    send_selection_email(row["email"], row.get("name", ""))
                else:
                    df.at[idx, "verdict"] = "Rejected"
                    send_rejection_email(row["email"], row.get("name", ""))

            # Sort and filter
            shortlisted = df[df["verdict"] == "Selected"].sort_values(by="score", ascending=False)
            if top_n:
                shortlisted = shortlisted.head(top_n)

            st.success("✅ Resume analysis and email dispatch complete!")
            st.dataframe(shortlisted if not shortlisted.empty else df)
