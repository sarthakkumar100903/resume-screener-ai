import smtplib
import logging
import time
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, List, Optional, Any
import pandas as pd
from constants import EMAIL_TEMPLATES
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class EmailConfig:
    """Email configuration settings"""
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    sender_email: str = "demoprojectid3@gmail.com"
    sender_password: str = "rpdsmhgbvppgldjx"  # App password from .env
    company_name: str = "EazyAI Technologies"
    use_ssl: bool = False  # Use STARTTLS instead
    timeout: int = 30

# Global email configuration - load from environment variables
email_config = EmailConfig(
    sender_email=os.getenv("SMTP_USER", "demoprojectid3@gmail.com"),
    sender_password=os.getenv("SMTP_PASS", "rpdsmhgbvppgldjx"),
    smtp_server=os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    smtp_port=int(os.getenv("SMTP_PORT", "587"))
)

def send_email(
    to_email: str, 
    subject: str, 
    body: str, 
    sender_name: str = "EazyAI Recruitment Team",
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
    
    # Check email configuration
    if not email_config.sender_email or not email_config.sender_password:
        logger.error("Email configuration missing - check SMTP_USER and SMTP_PASS environment variables")
        return False
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Attempting to send email to {to_email} (attempt {attempt + 1})")
            
            # Create message
            message = MIMEMultipart()
            message["From"] = f"{sender_name} <{email_config.sender_email}>"
            message["To"] = to_email
            message["Subject"] = subject
            
            # Add body
            message.attach(MIMEText(body, "plain"))
            
            # Send email using STARTTLS
            with smtplib.SMTP(email_config.smtp_server, email_config.smtp_port) as server:
                server.set_debuglevel(0)  # Set to 1 for debugging
                server.starttls()
                server.login(email_config.sender_email, email_config.sender_password)
                text = message.as_string()
                server.sendmail(email_config.sender_email, to_email, text)
            
            logger.info(f"Email sent successfully to {to_email}")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"SMTP authentication failed: {str(e)}")
            logger.error("Check your email credentials and app password")
            return False  # Don't retry authentication failures
            
        except smtplib.SMTPRecipientsRefused as e:
            logger.error(f"Recipient refused: {to_email} - {str(e)}")
            return False  # Don't retry recipient failures
            
        except smtplib.SMTPConnectError as e:
            logger.error(f"SMTP connection failed: {str(e)}")
            if attempt < max_retries - 1:
                logger.info(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Failed to connect after {max_retries} attempts")
                return False
                
        except (smtplib.SMTPException, ConnectionError, TimeoutError) as e:
            logger.warning(f"Email attempt {attempt + 1} failed for {to_email}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
            else:
                logger.error(f"Failed to send email to {to_email} after {max_retries} attempts")
                return False
        
        except Exception as e:
            logger.error(f"Unexpected error sending email to {to_email}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
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
    
    # Prepare template data with safe defaults
    template_data = {
        "name": safe_get_field(candidate_data, "name", "Candidate"),
        "role": safe_get_field(candidate_data, "jd_role", "the position"),
        "company_name": email_config.company_name,
        "highlights": format_highlights(candidate_data.get("highlights", [])),
    }
    
    # Add all candidate data safely
    for key, value in candidate_data.items():
        if key not in template_data:
            template_data[key] = safe_get_field(candidate_data, key, "N/A")
    
    # Add custom data if provided
    if custom_data:
        template_data.update(custom_data)
    
    try:
        # Format subject and body with error handling
        subject = template["subject"].format(**template_data)
        body = template["body"].format(**template_data)
        
        return send_email(to_email, subject, body)
        
    except KeyError as e:
        logger.error(f"Missing template variable: {str(e)}")
        logger.error(f"Available variables: {list(template_data.keys())}")
        return False
    except Exception as e:
        logger.error(f"Template formatting error: {str(e)}")
        return False

def send_bulk_emails(
    candidates: List[Dict[str, Any]],
    template_type: str,
    progress_callback: Optional[callable] = None,
    batch_size: int = 5,  # Reduced batch size for stability
    batch_delay: float = 3.0  # Increased delay between batches
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
        logger.warning("No candidates provided for bulk email")
        return results
    
    logger.info(f"Starting bulk email send: {len(candidates)} emails, template: {template_type}")
    
    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        batch_start_time = time.time()
        
        logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} emails")
        
        for j, candidate in enumerate(batch):
            candidate_name = safe_get_field(candidate, "name", "Unknown")
            email = safe_get_field(candidate, "email", "").strip()
            
            if not email or email in ['N/A', '', 'nan', 'None']:
                results["failed"] += 1
                results["errors"].append({
                    "candidate": candidate_name,
                    "error": "Missing or invalid email address"
                })
                logger.warning(f"No valid email for {candidate_name}")
                continue
            
            # Send email
            logger.info(f"Sending email to {candidate_name} ({email})")
            if send_templated_email(email, template_type, candidate):
                results["sent"] += 1
                logger.info(f"✅ Email sent to {candidate_name}")
            else:
                results["failed"] += 1
                results["errors"].append({
                    "candidate": candidate_name,
                    "email": email,
                    "error": "Failed to send email"
                })
                logger.error(f"❌ Failed to send email to {candidate_name}")
            
            # Update progress
            if progress_callback:
                progress = (i + j + 1) / len(candidates)
                progress_callback(progress, candidate_name)
            
            # Small delay between individual emails
            time.sleep(0.5)
        
        batch_time = time.time() - batch_start_time
        logger.info(f"Completed batch {i//batch_size + 1}: {len(batch)} emails in {batch_time:.2f}s")
        
        # Rate limiting between batches
        if i + batch_size < len(candidates):
            logger.info(f"Waiting {batch_delay}s before next batch...")
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
    if safe_get_score(candidate_data, "score") == 0:
        missing_fields.append("Valid Resume Content")
    
    if safe_get_score(candidate_data, "jd_similarity") < 10:
        missing_fields.append("Relevant Experience")
    
    # Check for empty fitment
    if is_missing_value(candidate_data.get("fitment")):
        missing_fields.append("Candidate Fitment Analysis")
    
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
    
    if not is_valid_email(to_email):
        logger.error(f"Invalid email address: {to_email}")
        return False
    
    missing_str = format_list_for_email(missing_fields)
    candidate_name = candidate_name if candidate_name and candidate_name != "N/A" else "Candidate"
    
    subject = f"Additional Information Required - {role} Application"
    
    body = f"""Dear {candidate_name},

Thank you for your interest in the {role} position at {email_config.company_name}.

We have received your application and are currently reviewing it. However, we noticed that the following information is missing or incomplete:

{missing_str}

To continue processing your application, please reply to this email with the missing information {deadline}.

If you have any questions about this request, please don't hesitate to contact us.

Best regards,
{email_config.company_name} Recruitment Team

---
Please reply to this email with the requested information. Do not reply if this message was sent in error."""
    
    return send_email(to_email, subject, body)

def send_interview_invitation(
    candidate_data: Dict[str, Any],
    interview_details: Dict[str, Any]
) -> bool:
    """
    Send interview invitation email with comprehensive details
    """
    required_details = ["date", "time", "format", "duration"]
    for detail in required_details:
        if detail not in interview_details:
            logger.error(f"Missing required interview detail: {detail}")
            return False
    
    email = safe_get_field(candidate_data, "email", "")
    if not email or not is_valid_email(email):
        logger.error("No valid email address for interview invitation")
        return False
    
    name = safe_get_field(candidate_data, "name", "Candidate")
    role = safe_get_field(candidate_data, "jd_role", "the position")
    
    subject = f"Interview Invitation - {role} Position at {email_config.company_name}"
    
    # Format interview details
    interview_format = interview_details["format"].title()
    meeting_link = interview_details.get("meeting_link", "")
    location = interview_details.get("location", "")
    interviewer = interview_details.get("interviewer", "Our team")
    
    body = f"""Dear {name},

Congratulations! We are pleased to invite you for an interview for the {role} position at {email_config.company_name}.

INTERVIEW DETAILS:
📅 Date: {interview_details['date']}
🕒 Time: {interview_details['time']}
💻 Format: {interview_format}
⏱️ Duration: {interview_details['duration']} minutes
👥 Interviewer: {interviewer}"""

    if meeting_link:
        body += f"\n🔗 Meeting Link: {meeting_link}"
    
    if location:
        body += f"\n📍 Location: {location}"
    
    body += f"""

Please confirm your availability by replying to this email at least 24 hours before the scheduled interview.

WHAT TO EXPECT:
• Technical discussion about your experience and skills
• Questions about your approach to problem-solving
• Overview of the role and our company culture
• Opportunity to ask questions about the position

PREPARATION:
• Review the job description and requirements
• Prepare examples of relevant projects and achievements
• Have questions ready about the role and company
• Ensure stable internet connection (for virtual interviews)

If you need to reschedule or have any questions, please contact us as soon as possible.

We look forward to meeting you and learning more about your experience!

Best regards,
{email_config.company_name} Recruitment Team

---
This is an interview invitation. Please respond to confirm your attendance."""
    
    return send_email(email, subject, body)

def send_status_update_email(
    candidate_data: Dict[str, Any],
    status: str,
    custom_message: str = ""
) -> bool:
    """
    Send status update email to candidate
    """
    status_mapping = {
        "shortlisted": "shortlist",
        "shortlist": "shortlist",
        "under_review": "review",
        "review": "review",
        "rejected": "reject",
        "reject": "reject"
    }
    
    template_type = status_mapping.get(status.lower())
    if not template_type:
        logger.error(f"Unknown status type: {status}")
        return False
    
    # Add custom message if provided
    custom_data = {}
    if custom_message:
        custom_data["custom_message"] = custom_message
    
    email = safe_get_field(candidate_data, "email", "")
    if not email or not is_valid_email(email):
        logger.error("Invalid email address for status update")
        return False
    
    return send_templated_email(email, template_type, candidate_data, custom_data)

# Utility functions

def is_valid_email(email: str) -> bool:
    """Validate email format with comprehensive check"""
    if not email or not isinstance(email, str):
        return False
    
    import re
    # More comprehensive email regex
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    
    # Basic format check
    if not re.match(pattern, email):
        return False
    
    # Additional checks
    if '..' in email or email.startswith('.') or email.endswith('.'):
        return False
    
    if '@' not in email or email.count('@') != 1:
        return False
    
    local, domain = email.split('@')
    if not local or not domain:
        return False
    
    return True

def is_missing_value(value: Any) -> bool:
    """Check if a value is considered missing"""
    if value is None:
        return True
    if pd.isna(value):
        return True
    if str(value).strip().lower() in ["", "n/a", "na", "null", "none", "nan"]:
        return True
    return False

def safe_get_field(data: dict, key: str, default: Any = "N/A") -> Any:
    """Safely get field from data with fallback"""
    value = data.get(key, default)
    if is_missing_value(value):
        return default
    return value

def safe_get_score(data: dict, key: str, default: int = 0) -> int:
    """Safely get numeric score from data"""
    value = data.get(key, default)
    try:
        return int(float(value)) if not is_missing_value(value) else default
    except (ValueError, TypeError):
        return default

def format_highlights(highlights: List[str]) -> str:
    """Format highlights list for email templates"""
    if not highlights or not isinstance(highlights, list):
        return "• Your qualifications and experience"
    
    # Filter out empty or invalid highlights
    valid_highlights = [h for h in highlights if h and str(h).strip() and str(h).strip() not in ["N/A", "n/a", "none"]]
    
    if not valid_highlights:
        return "• Your qualifications and experience"
    
    if len(valid_highlights) == 1:
        return f"• {valid_highlights[0]}"
    
    # Format up to 3 highlights
    formatted = []
    for i, highlight in enumerate(valid_highlights[:3]):
        formatted.append(f"• {highlight}")
    
    return "\n".join(formatted)

def format_list_for_email(items: List[str]) -> str:
    """Format a list of items for email body"""
    if not items:
        return "• No specific items identified"
    
    # Filter valid items
    valid_items = [item for item in items if item and str(item).strip()]
    
    if not valid_items:
        return "• No specific items identified"
    
    if len(valid_items) == 1:
        return f"• {valid_items[0]}"
    
    return "\n".join([f"• {item}" for item in valid_items])

def validate_email_config() -> bool:
    """Validate email configuration"""
    required_fields = ["sender_email", "sender_password", "smtp_server"]
    
    for field in required_fields:
        value = getattr(email_config, field, None)
        if not value:
            logger.error(f"Missing email configuration: {field}")
            return False
    
    # Test email format
    if not is_valid_email(email_config.sender_email):
        logger.error(f"Invalid sender email format: {email_config.sender_email}")
        return False
    
    logger.info("Email configuration validation passed")
    return True

def test_email_connection() -> bool:
    """Test email server connection"""
    if not validate_email_config():
        return False
    
    try:
        logger.info("Testing SMTP connection...")
        with smtplib.SMTP(email_config.smtp_server, email_config.smtp_port) as server:
            server.set_debuglevel(0)
            server.starttls()
            server.login(email_config.sender_email, email_config.sender_password)
        
        logger.info("✅ Email connection test successful")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"❌ Email authentication failed: {str(e)}")
        logger.error("Check your email credentials and app password")
        return False
    except smtplib.SMTPConnectError as e:
        logger.error(f"❌ SMTP connection failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Email connection test failed: {str(e)}")
        return False

def send_test_email(test_recipient: str = None) -> bool:
    """Send a test email to verify configuration"""
    if not test_recipient:
        test_recipient = email_config.sender_email
    
    if not is_valid_email(test_recipient):
        logger.error(f"Invalid test recipient email: {test_recipient}")
        return False
    
    subject = "Test Email - EazyAI Resume Screener"
    body = f"""This is a test email from EazyAI Resume Screener.

Configuration:
• SMTP Server: {email_config.smtp_server}:{email_config.smtp_port}
• Sender: {email_config.sender_email}
• Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}

If you received this email, the email configuration is working correctly.

Best regards,
EazyAI System"""
    
    logger.info(f"Sending test email to {test_recipient}")
    return send_email(test_recipient, subject, body)

# Email analytics and reporting

class EmailAnalytics:
    """Track email sending analytics with enhanced features"""
    
    def __init__(self):
        self.reset_stats()
    
    def reset_stats(self):
        """Reset all statistics"""
        self.total_sent = 0
        self.total_failed = 0
        self.template_usage = {}
        self.error_types = {}
        self.recipient_status = {}
        self.start_time = time.time()
        
        logger.info("Email analytics reset")
    
    def record_success(self, template_type: str = "unknown", recipient: str = ""):
        """Record successful email send"""
        self.total_sent += 1
        self.template_usage[template_type] = self.template_usage.get(template_type, 0) + 1
        
        if recipient:
            self.recipient_status[recipient] = "sent"
        
        logger.debug(f"Email success recorded: {template_type}")
    
    def record_failure(self, error_type: str = "unknown", recipient: str = ""):
        """Record failed email send"""
        self.total_failed += 1
        self.error_types[error_type] = self.error_types.get(error_type, 0) + 1
        
        if recipient:
            self.recipient_status[recipient] = f"failed: {error_type}"
        
        logger.debug(f"Email failure recorded: {error_type}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        total_attempts = self.total_sent + self.total_failed
        success_rate = (self.total_sent / total_attempts * 100) if total_attempts > 0 else 0
        elapsed_time = time.time() - self.start_time
        
        stats = {
            "total_sent": self.total_sent,
            "total_failed": self.total_failed,
            "total_attempts": total_attempts,
            "success_rate": success_rate,
            "elapsed_time": elapsed_time,
            "emails_per_minute": (total_attempts / elapsed_time * 60) if elapsed_time > 0 else 0,
            "template_usage": self.template_usage,
            "error_types": self.error_types,
            "most_common_error": max(self.error_types.items(), key=lambda x: x[1])[0] if self.error_types else "None"
        }
        
        return stats
    
    def print_summary(self):
        """Print analytics summary"""
        stats = self.get_stats()
        
        print("\n📧 EMAIL ANALYTICS SUMMARY")
        print("=" * 40)
        print(f"Total Sent: {stats['total_sent']}")
        print(f"Total Failed: {stats['total_failed']}")
        print(f"Success Rate: {stats['success_rate']:.1f}%")
        print(f"Emails per Minute: {stats['emails_per_minute']:.1f}")
        
        if stats['template_usage']:
            print("\nTemplate Usage:")
            for template, count in stats['template_usage'].items():
                print(f"  {template}: {count}")
        
        if stats['error_types']:
            print("\nError Types:")
            for error, count in stats['error_types'].items():
                print(f"  {error}: {count}")
        
        print("=" * 40)

# Enhanced email queue system for batch processing
class EmailQueue:
    """Manage email queue with priority and retry logic"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 60.0):
        self.queue = []
        self.failed_queue = []
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.analytics = EmailAnalytics()
    
    def add_email(self, to_email: str, subject: str, body: str, priority: int = 1):
        """Add email to queue with priority (1=high, 2=normal, 3=low)"""
        email_item = {
            "to_email": to_email,
            "subject": subject,
            "body": body,
            "priority": priority,
            "attempts": 0,
            "created_at": time.time()
        }
        
        self.queue.append(email_item)
        # Sort by priority (lower number = higher priority)
        self.queue.sort(key=lambda x: x["priority"])
        
        logger.info(f"Email added to queue: {to_email} (priority {priority})")
    
    def process_queue(self, batch_size: int = 5, batch_delay: float = 2.0) -> Dict[str, int]:
        """Process email queue in batches"""
        results = {"sent": 0, "failed": 0, "skipped": 0}
        
        logger.info(f"Processing email queue: {len(self.queue)} emails")
        
        while self.queue:
            # Process batch
            batch = self.queue[:batch_size]
            self.queue = self.queue[batch_size:]
            
            for email_item in batch:
                email_item["attempts"] += 1
                
                if send_email(email_item["to_email"], email_item["subject"], email_item["body"]):
                    results["sent"] += 1
                    self.analytics.record_success("queued", email_item["to_email"])
                    logger.info(f"✅ Queued email sent: {email_item['to_email']}")
                else:
                    if email_item["attempts"] < self.max_retries:
                        # Add back to failed queue for retry
                        self.failed_queue.append(email_item)
                        logger.warning(f"Email failed, will retry: {email_item['to_email']}")
                    else:
                        results["failed"] += 1
                        self.analytics.record_failure("max_retries_reached", email_item["to_email"])
                        logger.error(f"❌ Email failed permanently: {email_item['to_email']}")
            
            # Delay between batches
            if self.queue:  # Only delay if more emails to process
                logger.info(f"Waiting {batch_delay}s before next batch...")
                time.sleep(batch_delay)
        
        # Process failed queue if any
        if self.failed_queue:
            logger.info(f"Processing {len(self.failed_queue)} failed emails for retry...")
            time.sleep(self.retry_delay)
            
            retry_queue = self.failed_queue.copy()
            self.failed_queue.clear()
            
            for email_item in retry_queue:
                if email_item["attempts"] < self.max_retries:
                    self.queue.append(email_item)
        
        logger.info(f"Queue processing complete: {results}")
        return results
    
    def get_queue_status(self) -> Dict[str, int]:
        """Get current queue status"""
        return {
            "pending": len(self.queue),
            "failed_retry": len(self.failed_queue),
            "total": len(self.queue) + len(self.failed_queue)
        }

# Global instances
email_analytics = EmailAnalytics()
email_queue = EmailQueue()

# Initialization function
def initialize_email_system() -> bool:
    """Initialize email system and validate configuration"""
    logger.info("Initializing email system...")
    
    # Validate configuration
    if not validate_email_config():
        logger.error("Email system initialization failed: Invalid configuration")
        return False
    
    # Test connection
    if not test_email_connection():
        logger.warning("Email system initialized with connection issues")
        return False
    
    logger.info("✅ Email system initialized successfully")
    return True

# Auto-initialize on import
if __name__ != "__main__":
    try:
        initialize_email_system()
    except Exception as e:
        logger.error(f"Email system auto-initialization failed: {str(e)}")
