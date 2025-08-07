import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

FROM_EMAIL = os.getenv("EMAIL")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465

def send_email(to_email, subject, body):
    message = MIMEMultipart()
    message["From"] = FROM_EMAIL
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(FROM_EMAIL, EMAIL_PASSWORD)
            server.sendmail(FROM_EMAIL, to_email, message.as_string())
        print(f"✅ Email sent to {to_email} with subject: {subject}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")
        return False

def check_missing_info(row):
    missing_fields = []

    def is_missing(value):
        return pd.isna(value) or str(value).strip().lower() in ["", "n/a"]

    if is_missing(row.get("name", "")):
        missing_fields.append("name")
    if is_missing(row.get("email", "")):
        missing_fields.append("email")
    if is_missing(row.get("phone", "")):
        missing_fields.append("phone number")

    return missing_fields

def send_missing_info_email(to_email, name, missing_fields):
    subject = "Additional Information Required for Application"
    missing_str = ", ".join(missing_fields).title()
    body = f"""
    Dear {name or 'Candidate'},

    Thank you for submitting your resume for the position.

    However, we noticed that the following information is missing from your application: {missing_str}.
    Kindly reply to this email with the missing details so we can continue processing your application.

    Best regards,
    Recruitment Team
    """
    return send_email(to_email, subject, body)

def send_rejection_email(to_email, name):
    subject = "Update on Your Job Application"
    body = f"""
    Dear {name or 'Candidate'},

    Thank you for your interest in the position and for taking the time to apply.

    After careful consideration, we regret to inform you that we will not be moving forward with your application at this time.

    We wish you all the best in your job search and future endeavors.

    Best regards,
    Recruitment Team
    """
    return send_email(to_email, subject, body)

def send_selection_email(to_email, name):
    subject = "Congratulations! You've Been Selected"
    body = f"""
    Dear {name or 'Candidate'},

    We are pleased to inform you that you have been selected for the next stage of our hiring process.

    Our team will be in touch shortly with the next steps. Congratulations once again!

    Best regards,
    Recruitment Team
    """
    return send_email(to_email, subject, body)

# ✅ Helper function to send appropriate email based on verdict
def send_email_to_candidate(row):
    email = row.get("email", "")
    name = row.get("name", "")
    verdict = row.get("verdict", "").lower()

    if not email:
        print("❌ No email found for candidate.")
        return

    missing_fields = check_missing_info(row)
    if missing_fields:
        send_missing_info_email(email, name, missing_fields)
    elif verdict == "selected":
        send_selection_email(email, name)
    elif verdict == "rejected":
        send_rejection_email(email, name)
