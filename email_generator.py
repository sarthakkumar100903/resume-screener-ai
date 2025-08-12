import smtplib
import logging
import time
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
import pandas as pd
from constants import EMAIL_TEMPLATES, AZURE_CONFIG
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EmailConfig:
    """Email configuration settings"""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = "demoprojectid3@gmail.com"
    sender_password: str = "rpdsmhgbvppgldjx"  # App password
    company_name: str = "EazyAI Technologies"
    use_ssl: bool = True
    timeout: int = 30

# Global email configuration
email_config = EmailConfig()

def send_email(
    to_email: str, 
    subject: str, 
    body: str, 
    sender_name: str = "Recruitment Team",
    max_retries: int = 3,
    retry_delay: float = 1.0
) -> bool:
    """
    Enhanced email sending with retry logic and better error handling
    """
    if not to_email or not to_email.strip():
        logger.error("Invalid recipient email address")
        return False
    
    if not subject or not body:
        logger.error("Subject and body are required")
        return False
    
    # Clean and validate email
    to_email = to_email.strip().lower()
    if not is_valid_email(to_email):
        logger.error(f"Invalid email format: {to_email}")
        return False
    
    for attempt in range(max_retries):
        try:
            # Create message
            message = MIMEMultipart()
            message["From"] = f"{sender_name} <{email_config.sender_email}>"
            message["To"] = to_email
            message["Subject"] = subject
            
            # Add body
            message.attach(MIMEText(body, "plain"))
            
            # Send email
            if email_config.use_ssl:
                with smtplib.SMTP_SSL(email_config.smtp_server, 465) as server:
                    server.login(email_config.sender_email, email_config.sender_password)
                    text = message.as_string()
                    server.sendmail(email_config.sender_email, to_email, text)
            else:
                with smtplib.SMTP(email_config.smtp_server, email_config.smtp_port) as server:
                    server.starttls()
                    server.login(email_config.sender_email, email_config.sender_password)
                    text = message.as_string()
                    server.sendmail(email_config.sender_email, to_email, text)
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {str(e)}")
            return False  # Don't retry authentication failures
            
        except smtplib.SMTPRecipientsRefused as e:
            logger.error(f"Recipient refused: {to_email} - {str(e)}")
            return False  # Don't retry recipient failures
            
        except (smtplib.SMTPException, ConnectionError) as e:
            logger.warning(f"Email attempt {attempt + 1} failed for {to_email}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Failed to send email to {to_email} after {max_retries} attempts")
                return False
        
        except Exception as e:
            logger.error(f"Unexpected error sending email to {to_email}: {str(e)}")
            return False
    
    return False

def send_templated_email(
    to_email: str,
    template_type: str,
    candidate_data: Dict[str, Any],
    custom_data: Dict[str, Any] = None
) -> bool:
    """
    Send email using predefined templates with data substitution
    """
    if template_type not in EMAIL_TEMPLATES:
        logger.error(f"Unknown template type: {template_type}")
        return False
    
    template = EMAIL_TEMPLATES[template_type]
    
    # Prepare template data
    template_data = {
        "name": candidate_data.get("name", "Candidate"),
        "role": candidate_data.get("jd_role", "the position"),
        "company_name": email_config.company_name,
        "highlights": format_highlights(candidate_data.get("highlights", [])),
        **candidate_data
    }
    
    # Add custom data if provided
    if custom_data:
        template_data.update(custom_data)
    
    try:
        # Format subject and body
        subject = template["subject"].format(**template_data)
        body = template["body"].format(**template_data)
        
        return send_email(to_email, subject, body)
        
    except KeyError as e:
        logger.error(f"Missing template variable: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Template formatting error: {str(e)}")
        return False

def send_bulk_emails(
    candidates: List[Dict[str, Any]],
    template_type: str,
    progress_callback: Optional[callable] = None,
    batch_size: int = 10,
    batch_delay: float = 2.0
) -> Dict[str, Any]:
    """
    Send bulk emails with progress tracking and rate limiting
    """
    results = {
        "total": len(candidates),
        "sent": 0,
        "failed": 0,
        "errors": []
    }
    
    if not candidates:
        return results
    
    logger.info(f"Starting bulk email send: {len(candidates)} emails, template: {template_type}")
    
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        batch_start_time = time.time()
        
        for j, candidate in enumerate(batch):
            email = candidate.get("email", "").strip()
            if not email or email == "N/A":
                results["failed"] += 1
                results["errors"].append({
                    "candidate": candidate.get("name", "Unknown"),
                    "error": "Missing email address"
                })
                continue
            
            # Send email
            if send_templated_email(email, template_type, candidate):
                results["sent"] += 1
                logger.debug(f"Email sent to {candidate.get('name', 'Unknown')} ({email})")
            else:
                results["failed"] += 1
                results["errors"].append({
                    "candidate": candidate.get("name", "Unknown"),
                    "email": email,
                    "error": "Failed to send email"
                })
            
            # Update progress
            if progress_callback:
                progress = (i + j + 1) / len(candidates)
                progress_callback(progress, candidate.get("name", "Unknown"))
        
        batch_time = time.time() - batch_start_time
        logger.info(f"Processed batch {i//batch_size + 1}: {len(batch)} emails in {batch_time:.2f}s")
        
        # Rate limiting between batches
        if i + batch_size < len(candidates):
            time.sleep(batch_delay)
    
    success_rate = (results["sent"] / results["total"] * 100) if results["total"] > 0 else 0
    logger.info(f"Bulk email completed: {results['sent']}/{results['total']} sent ({success_rate:.1f}% success rate)")
    
    return results

def check_missing_info(candidate_data: Dict[str, Any]) -> List[str]:
    """
    Enhanced function to check for missing candidate information
    """
    missing_fields = []
    
    # Define required fields and their display names
    required_fields = {
        "name": "Full Name",
        "email": "Email Address",
        "phone": "Phone Number"
    }
    
    for field, display_name in required_fields.items():
        value = candidate_data.get(field, "")
        if is_missing_value(value):
            missing_fields.append(display_name)
    
    # Check for additional quality indicators
    if candidate_data.get("score", 0) == 0:
        missing_fields.append("Valid Resume Content")
    
    if candidate_data.get("jd_similarity", 0) < 10:
        missing_fields.append("Relevant Experience")
    
    return missing_fields

def send_missing_info_email(
    to_email: str, 
    candidate_name: str, 
    missing_fields: List[str],
    role: str = "the position",
    deadline: str = "within 3 business days"
) -> bool:
    """
    Enhanced function to send missing information request emails
    """
    if not missing_fields:
        logger.warning("No missing fields specified for missing info email")
        return False
    
    missing_str = format_list_for_email(missing_fields)
    
    subject = f"Additional Information Required - {role} Application"
    
    body = f"""Dear {candidate_name or 'Candidate'},

Thank you for your interest in the {role} position at {email_config.company_name}.

We have received your application and are currently reviewing it. However, we noticed that the following information is missing or incomplete:

{missing_str}

To continue processing your application, please reply to this email with the missing information {deadline}.

If you have any questions about this request, please don't hesitate to contact us.

Best regards,
{email_config.company_name} Recruitment Team

---
This is an automated message. Please do not reply to this email address if it's marked as no-reply."""
    
    return send_email(to_email, subject, body)

def send_interview_invitation(
    candidate_data: Dict[str, Any],
    interview_details: Dict[str, Any]
) -> bool:
    """
    Send interview invitation email
    """
    required_details = ["date", "time", "format", "duration"]
    if not all(key in interview_details for key in required_details):
        logger.error("Missing required interview details")
        return False
    
    email = candidate_data.get("email", "")
    if not email or email == "N/A":
        logger.error("No valid email address for interview invitation")
        return False
    
    name = candidate_data.get("name", "Candidate")
    role = candidate_data.get("jd_role", "the position")
    
    subject = f"Interview Invitation - {role} Position"
    
    # Format interview details
    interview_format = interview_details["format"].title()
    meeting_link = interview_details.get("meeting_link", "")
    location = interview_details.get("location", "")
    
    body = f"""Dear {name},

Congratulations! We are pleased to invite you for an interview for the {role} position at {email_config.company_name}.

INTERVIEW DETAILS:
• Date: {interview_details['date']}
• Time: {interview_details['time']}
• Format: {interview_format}
• Duration: {interview_details['duration']} minutes"""

    if meeting_link:
        body += f"\n• Meeting Link: {meeting_link}"
    
    if location:
        body += f"\n• Location: {location}"
    
    body += f"""

Please confirm your availability by replying to this email at least 24 hours before the scheduled interview.

What to expect:
• Technical discussion about your experience
• Questions about your approach to problem-solving
• Opportunity to ask questions about the role and company

If you need to reschedule, please contact us as soon as possible.

We look forward to meeting you!

Best regards,
{email_config.company_name} Recruitment Team"""
    
    return send_email(email, subject, body)

def send_status_update_email(
    candidate_data: Dict[str, Any],
    status: str,
    custom_message: str = ""
) -> bool:
    """
    Send status update email to candidate
    """
    status_templates = {
        "shortlisted": "shortlist",
        "under_review": "review", 
        "rejected": "reject",
        "interview_scheduled": "interview",
        "offer_extended": "offer"
    }
    
    template_type = status_templates.get(status.lower())
    if not template_type:
        logger.error(f"Unknown status type: {status}")
        return False
    
    # Add custom message if provided
    custom_data = {}
    if custom_message:
        custom_data["custom_message"] = custom_message
    
    return send_templated_email(
        candidate_data.get("email", ""),
        template_type,
        candidate_data,
        custom_data
    )

# Utility functions

def is_valid_email(email: str) -> bool:
    """Validate email format"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

def is_missing_value(value: Any) -> bool:
    """Check if a value is considered missing"""
    if value is None:
        return True
    if pd.isna(value):
        return True
    if str(value).strip().lower() in ["", "n/a", "na", "null", "none"]:
        return True
    return False

def format_highlights(highlights: List[str]) -> str:
    """Format highlights list for email templates"""
    if not highlights or not isinstance(highlights, list):
        return "Your qualifications and experience"
    
    if len(highlights) == 1:
        return highlights[0]
    
    formatted = []
    for i, highlight in enumerate(highlights[:3]):  # Limit to top 3
        formatted.append(f"• {highlight}")
    
    return "\n".join(formatted)

def format_list_for_email(items: List[str]) -> str:
    """Format a list of items for email body"""
    if not items:
        return "No items specified"
    
    if len(items) == 1:
        return f"• {items[0]}"
    
    return "\n".join([f"• {item}" for item in items])

def validate_email_config() -> bool:
    """Validate email configuration"""
    required_fields = ["sender_email", "sender_password", "smtp_server"]
    
    for field in required_fields:
        if not getattr(email_config, field):
            logger.error(f"Missing email configuration: {field}")
            return False
    
    return True

def test_email_connection() -> bool:
    """Test email server connection"""
    try:
        if email_config.use_ssl:
            with smtplib.SMTP_SSL(email_config.smtp_server, 465) as server:
                server.login(email_config.sender_email, email_config.sender_password)
        else:
            with smtplib.SMTP(email_config.smtp_server, email_config.smtp_port) as server:
                server.starttls()
                server.login(email_config.sender_email, email_config.sender_password)
        
        logger.info("Email connection test successful")
        return True
        
    except Exception as e:
        logger.error(f"Email connection test failed: {str(e)}")
        return False

# Email analytics and reporting

class EmailAnalytics:
    """Track email sending analytics"""
    
    def __init__(self):
        self.reset_stats()
    
    def reset_stats(self):
        """Reset all statistics"""
        self.total_sent = 0
        self.total_failed = 0
        self.template_usage = {}
        self.error_types = {}
        self.start_time = time.time()
    
    def record_success(self, template_type: str = "unknown"):
        """Record successful email send"""
        self.total_sent += 1
        self.template_usage[template_type] = self.template_usage.get(template_type, 0) + 1
    
    def record_failure(self, error_type: str = "unknown"):
        """Record failed email send"""
        self.total_failed += 1
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        total_attempts = self.total_sent + self.total_failed
        success_rate = (self.total_sent / total_attempts * 100) if total_attempts > 0 else 0
        elapsed_time = time.time() - self.start_time
        
        return {
            "total_sent": self.total_sent,
            "total_failed": self.total_failed,
            "total_attempts": total_attempts,
            "success_rate": success_rate,
            "elapsed_time": elapsed_time,
            "emails_per_minute": (total_attempts / elapsed_time * 60) if elapsed_time > 0 else 0,
            "template_usage": self.template_usage,
            "error_types": self.error_types
        }

# Global analytics instance
email_analytics = EmailAnalytics()
