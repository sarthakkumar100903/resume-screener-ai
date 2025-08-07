import streamlit as st
import os
import pandas as pd
from backend import analyze_resumes_from_blob
from email_generator import send_email_to_candidate

# Threshold sliders
st.sidebar.title("Resume Filter Criteria")
jd_thresh = st.sidebar.slider("JD Match Score Threshold", 0, 100, 50)
skills_thresh = st.sidebar.slider("Skills Match Threshold", 0, 100, 50)
domain_thresh = st.sidebar.slider("Domain Match Threshold", 0, 100, 50)
exp_thresh = st.sidebar.slider("Experience Match Threshold", 0, 100, 50)
top_n = st.sidebar.number_input("Top N candidates to shortlist", min_value=1, value=5)

st.title("AI Resume Screener")

# Analyze resumes from Azure Blob
df = analyze_resumes_from_blob()

if df.empty:
    st.warning("⚠️ No resumes found in storage or analysis failed.")
else:
    st.success("✅ Resumes analyzed successfully!")

    # Show available columns (debugging aid)
    st.write("📊 Columns in DataFrame:", df.columns.tolist())

    # Warn if score columns are missing
    required_cols = ["jd_score", "skills_score", "domain_score", "experience_score"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.warning(f"⚠️ Missing score columns in data: {missing_cols}")

    # Verdict logic (safe)
    def verdict_logic(row):
        try:
            if (
                row.get("jd_score", 0) >= jd_thresh and
                row.get("skills_score", 0) >= skills_thresh and
                row.get("domain_score", 0) >= domain_thresh and
                row.get("experience_score", 0) >= exp_thresh
            ):
                return "selected"
            else:
                return "rejected"
        except Exception as e:
            print(f"Error in verdict_logic: {e}")
            return "rejected"

    # Apply verdict logic
    df["verdict"] = df.apply(verdict_logic, axis=1)

    # Handle missing emails
    df["email"] = df["email"].fillna("")

    # Show results table
    st.subheader("Analyzed Resume Data")
    st.dataframe(df)

    # Send emails based on verdict
    for _, row in df.iterrows():
        email = row["email"]
        name = row.get("name", "Candidate")
        verdict = row["verdict"]

        if email:  # Only send if email is present
            if verdict == "selected":
                send_email_to_candidate(email, "selection", name)
            elif verdict == "rejected":
                send_email_to_candidate(email, "rejection", name)
            elif verdict == "missing_info":
                send_email_to_candidate(email, "missing_info", name)
