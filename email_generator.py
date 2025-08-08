import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd

def send_email(to_email, subject, body):
    from_email = "demoprojectid3@gmail.com"
    password = "rpdsmhgbvppgldjx"  # Gmail App Password

    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(from_email, password)
            server.sendmail(from_email, to_email, message.as_string())
        return True
    except Exception as e:
        print("Failed to send email: ", e)
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

def send_selection_email(to_email, name):
    subject = "Congratulations! You have been shortlisted"
    body = f"""
    Dear {name or 'Candidate'},

    Congratulations! You have been shortlisted for the next stage of the recruitment process.

    Our HR team will reach out to you with the interview details shortly.

    Best regards,  
    Recruitment Team
    """
    return send_email(to_email, subject, body)

def send_rejection_email(to_email, name):
    subject = "Application Status - Not Shortlisted"
    body = f"""
    Dear {name or 'Candidate'},

    Thank you for applying for the position.

    After careful consideration, we regret to inform you that you have not been shortlisted for the next stage.

    We wish you the best for your future endeavors.

    Best regards,  
    Recruitment Team
    """
    return send_email(to_email, subject, body)

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
