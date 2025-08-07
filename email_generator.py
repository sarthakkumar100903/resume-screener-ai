import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_email_to_candidate(to_email, verdict, candidate_name="Candidate"):
    from_email = "your_email@gmail.com"  # 🔁 Replace with your Gmail address
    app_password = "your_app_password_here"  # 🔁 Use App Password (not your Gmail password)

    if verdict == "selection":
        subject = "Congratulations! You've been shortlisted 🎉"
        body = f"""Dear {candidate_name},

We are pleased to inform you that you have been shortlisted for the next round of interviews.

Our HR team will connect with you shortly with further details.

Best regards,  
HR Team"""
    
    elif verdict == "rejection":
        subject = "Application Status Update"
        body = f"""Dear {candidate_name},

Thank you for taking the time to apply.

After careful review, we regret to inform you that you have not been selected for the position.

We appreciate your interest and wish you all the best in your career.

Warm regards,  
HR Team"""
    
    elif verdict == "missing_info":
        subject = "Incomplete Application - Action Required"
        body = f"""Dear {candidate_name},

We noticed that some required information is missing from your application.

Kindly resubmit your resume with complete details so we can process your application further.

Thanks and regards,  
HR Team"""
    
    else:
        print(f"⚠️ Unknown verdict '{verdict}' — skipping email to {to_email}")
        return  # Skip if verdict is unknown

    # Build the email message
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(from_email, app_password)
        server.send_message(msg)
        server.quit()
        print(f"✅ Email sent to {to_email} for verdict: {verdict}")
    except Exception as e:
        print(f"❌ Failed to send email to {to_email}: {e}")
