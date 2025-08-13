# pdf_utils.py — Generate candidate summaries as PDF with enhanced error handling

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.colors import HexColor, black, blue, green, red, orange
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from io import BytesIO
import logging
import pandas as pd
from typing import Dict, Any, List

# Configure logging
logger = logging.getLogger(__name__)

def safe_get_value(data: Dict[str, Any], key: str, default: Any = "N/A") -> str:
    """Safely get value from data dictionary with proper fallbacks"""
    value = data.get(key, default)
    
    # Handle pandas NaN values
    if pd.isna(value):
        return str(default)
    
    # Handle None values
    if value is None:
        return str(default)
    
    # Handle empty strings
    if isinstance(value, str) and not value.strip():
        return str(default)
    
    # Handle common "empty" values
    if str(value).strip().lower() in ["n/a", "na", "none", "null", "nan"]:
        return str(default)
    
    return str(value).strip()

def safe_get_score(data: Dict[str, Any], key: str, default: int = 0) -> int:
    """Safely get numeric score from data"""
    value = data.get(key, default)
    try:
        if pd.isna(value) or value is None:
            return default
        return int(float(value))
    except (ValueError, TypeError):
        return default

def safe_get_list(data: Dict[str, Any], key: str, default: List = None) -> List[str]:
    """Safely get list from data with proper handling"""
    if default is None:
        default = []
    
    value = data.get(key, default)
    
    if not value or pd.isna(value):
        return default
    
    if isinstance(value, list):
        # Filter out empty or invalid items
        return [str(item).strip() for item in value if item and str(item).strip() and str(item).strip().lower() not in ["n/a", "none", "null"]]
    
    if isinstance(value, str):
        # Handle string representations of lists
        if value.strip().lower() in ["n/a", "none", "null", "[]"]:
            return default
        # Try to split by common delimiters
        for delimiter in [';', ',', '\n', '|']:
            if delimiter in value:
                items = [item.strip() for item in value.split(delimiter) if item.strip()]
                return items if items else default
        return [value.strip()] if value.strip() else default
    
    return default

def wrap_text(text: str, max_width: int = 80) -> List[str]:
    """Wrap text to fit within specified width"""
    if not text or len(text) <= max_width:
        return [text]
    
    words = text.split()
    lines = []
    current_line = ""
    
    for word in words:
        if len(current_line + " " + word) <= max_width:
            current_line += " " + word if current_line else word
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    
    if current_line:
        lines.append(current_line)
    
    return lines

def generate_summary_pdf(candidate: Dict[str, Any]) -> bytes:
    """Generate comprehensive candidate summary as PDF with enhanced error handling"""
    
    try:
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        
        # Colors
        header_color = HexColor('#1e40af')  # Blue
        success_color = HexColor('#16a34a')  # Green
        warning_color = HexColor('#ea580c')  # Orange
        danger_color = HexColor('#dc2626')   # Red
        
        # Helper function to draw text with error handling
        def draw_text_safe(label, value, y_offset, icon="•", label_color=black, value_color=black):
            try:
                # Ensure values are strings and handle None/NaN
                safe_label = str(label) if label is not None else ""
                safe_value = safe_get_value(candidate, value if isinstance(value, str) else "", value if not isinstance(value, str) else "N/A")
                
                c.setFillColor(label_color)
                c.setFont("Helvetica-Bold", 10)
                c.drawString(50, y_offset, f"{icon} {safe_label}:")
                
                c.setFillColor(value_color)
                c.setFont("Helvetica", 10)
                
                # Handle long text by wrapping
                if len(safe_value) > 60:
                    wrapped_lines = wrap_text(safe_value, 60)
                    for i, line in enumerate(wrapped_lines[:3]):  # Limit to 3 lines
                        c.drawString(200, y_offset - (i * 12), line)
                    if len(wrapped_lines) > 3:
                        c.drawString(200, y_offset - (3 * 12), "...")
                else:
                    c.drawString(200, y_offset, safe_value)
                    
                c.setFillColor(black)  # Reset color
                return y_offset
            except Exception as e:
                logger.error(f"Error drawing text for {label}: {str(e)}")
                c.setFillColor(black)
                c.setFont("Helvetica", 10)
                c.drawString(200, y_offset, "Error displaying data")
                return y_offset

        # Start position
        y = height - inch
        spacing = 20
        
        # ======= Header ========
        try:
            c.setFillColor(header_color)
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, y, "📄 Candidate Assessment Report")
            c.setFillColor(black)
            y -= spacing * 1.5
            
            # Add generation timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.setFont("Helvetica", 8)
            c.drawString(50, y, f"Generated: {timestamp}")
            y -= spacing * 1.5
        except Exception as e:
            logger.error(f"Error creating header: {str(e)}")
            y -= spacing * 2
        
        # ======= Personal Information ========
        try:
            c.setFillColor(header_color)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "👤 Personal Information")
            c.setFillColor(black)
            y -= spacing
            
            draw_text_safe("Name", safe_get_value(candidate, "name"), y, "•")
            y -= spacing
            draw_text_safe("Email", safe_get_value(candidate, "email"), y, "📧")
            y -= spacing
            draw_text_safe("Phone", safe_get_value(candidate, "phone"), y, "📞")
            y -= spacing
            draw_text_safe("Applied Role", safe_get_value(candidate, "jd_role"), y, "💼")
            y -= spacing * 1.5
        except Exception as e:
            logger.error(f"Error creating personal info section: {str(e)}")
            y -= spacing * 3
        
        # ======= Scoring Overview ========
        try:
            c.setFillColor(header_color)
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, "📊 Assessment Scores")
            c.setFillColor(black)
            y -= spacing
            
            # Score mapping with colors
            scores = {
                "Overall Score": (safe_get_score(candidate, "score"), success_color if safe_get_score(candidate, "score") >= 70 else warning_color if safe_get_score(candidate, "score") >= 50 else danger_color),
                "JD Similarity": (safe_get_score(candidate, "jd_similarity"), black),
                "Skills Match": (safe_get_score(candidate, "skills_match"), black),
                "Domain Match": (safe_get_score(candidate, "domain_match"), black),
                "Experience Match": (safe_get_score(candidate, "experience_match"), black)
            }
            
            for score_name, (score_value, color) in scores.items():
                c.setFont("Helvetica-Bold", 10)
                c.setFillColor(black)
                c.drawString(50, y, f"• {score_name}:")
                
                c.setFont("Helvetica-Bold", 10)
                c.setFillColor(color)
                c.drawString(200, y, f"{score_value}%")
                
                # Add visual bar
                bar_width = min(score_value * 2, 200)  # Max 200 points width
                if bar_width > 0:
                    c.setFillColor(color)
                    c.rect(250, y - 2, bar_width, 8, fill=1)
                
                c.setFillColor(black)
                y -= spacing
            
            y -= spacing / 2
        except Exception as e:
            logger.error(f"Error creating scores section: {str(e)}")
            y -= spacing * 4
        
        # ======= Verdict with Color Coding ========
        try:
            verdict = safe_get_value(candidate, "verdict", "review").lower()
            verdict_colors = {
                "shortlist": success_color,
                "review": warning_color,
                "reject": danger_color
            }
            verdict_icons = {
                "shortlist": "✅",
                "review": "🔄",
                "reject": "❌"
            }
            
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(black)
            c.drawString(50, y, "📋 Final Verdict:")
            
            c.setFont("Helvetica-Bold", 14)
            c.setFillColor(verdict_colors.get(verdict, black))
            verdict_text = f"{verdict_icons.get(verdict, '•')} {verdict.upper()}"
            c.drawString(200, y, verdict_text)
            c.setFillColor(black)
            
            y -= spacing * 1.5
        except Exception as e:
            logger.error(f"Error creating verdict section: {str(e)}")
            y -= spacing * 2
        
        # Check if we need a new page
        if y < 300:  # Start new page if less than 300 points remaining
            c.showPage()
            y = height - inch
        
        # ======= Fitment Analysis ========
        try:
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(header_color)
            c.drawString(50, y, "🎯 Fitment Analysis")
            c.setFillColor(black)
            y -= spacing
            
            fitment_text = safe_get_value(candidate, "fitment", "Analysis not available")
            c.setFont("Helvetica", 10)
            
            # Wrap long fitment text
            fitment_lines = wrap_text(fitment_text, 90)
            for line in fitment_lines[:6]:  # Limit to 6 lines
                c.drawString(50, y, line)
                y -= spacing * 0.7
            
            if len(fitment_lines) > 6:
                c.drawString(50, y, "[Analysis truncated...]")
                y -= spacing * 0.7
            
            y -= spacing / 2
        except Exception as e:
            logger.error(f"Error creating fitment section: {str(e)}")
            y -= spacing * 3
        
        # ======= Summary ========
        try:
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(header_color)
            c.drawString(50, y, "📝 Candidate Summary")
            c.setFillColor(black)
            y -= spacing
            
            summary_text = safe_get_value(candidate, "summary_5_lines", "Summary not available")
            c.setFont("Helvetica", 10)
            
            # Wrap summary text
            summary_lines = wrap_text(summary_text, 90)
            for line in summary_lines[:8]:  # Limit to 8 lines
                c.drawString(50, y, line)
                y -= spacing * 0.7
            
            if len(summary_lines) > 8:
                c.drawString(50, y, "[Summary truncated...]")
                y -= spacing * 0.7
            
            y -= spacing / 2
        except Exception as e:
            logger.error(f"Error creating summary section: {str(e)}")
            y -= spacing * 4
        
        # Check if we need a new page for remaining sections
        if y < 400:  # Start new page if less than 400 points remaining
            c.showPage()
            y = height - inch
        
        # ======= Highlights ========
        try:
            highlights = safe_get_list(candidate, "highlights")
            if highlights:
                c.setFont("Helvetica-Bold", 12)
                c.setFillColor(success_color)
                c.drawString(50, y, "🌟 Key Highlights")
                c.setFillColor(black)
                y -= spacing
                
                c.setFont("Helvetica", 10)
                for highlight in highlights[:8]:  # Limit to 8 highlights
                    highlight_lines = wrap_text(f"• {highlight}", 85)
                    for line in highlight_lines[:2]:  # Max 2 lines per highlight
                        c.drawString(50, y, line)
                        y -= spacing * 0.7
                
                if len(highlights) > 8:
                    c.drawString(50, y, f"... and {len(highlights) - 8} more highlights")
                    y -= spacing * 0.7
                
                y -= spacing / 2
        except Exception as e:
            logger.error(f"Error creating highlights section: {str(e)}")
            y -= spacing * 2
        
        # ======= Red Flags (if any) ========
        try:
            red_flags = safe_get_list(candidate, "red_flags")
            if red_flags:
                c.setFont("Helvetica-Bold", 12)
                c.setFillColor(danger_color)
                c.drawString(50, y, "🚩 Red Flags")
                c.setFillColor(black)
                y -= spacing
                
                c.setFont("Helvetica", 10)
                for flag in red_flags[:6]:  # Limit to 6 red flags
                    flag_lines = wrap_text(f"• {flag}", 85)
                    for line in flag_lines[:2]:  # Max 2 lines per flag
                        c.drawString(50, y, line)
                        y -= spacing * 0.7
                
                y -= spacing / 2
        except Exception as e:
            logger.error(f"Error creating red flags section: {str(e)}")
            y -= spacing * 2
        
        # ======= Missing Information ========
        try:
            missing_gaps = safe_get_list(candidate, "missing_gaps")
            if missing_gaps:
                c.setFont("Helvetica-Bold", 12)
                c.setFillColor(warning_color)
                c.drawString(50, y, "❓ Missing Information")
                c.setFillColor(black)
                y -= spacing
                
                c.setFont("Helvetica", 10)
                for gap in missing_gaps[:6]:  # Limit to 6 gaps
                    gap_lines = wrap_text(f"• {gap}", 85)
                    for line in gap_lines[:2]:  # Max 2 lines per gap
                        c.drawString(50, y, line)
                        y -= spacing * 0.7
                
                y -= spacing / 2
        except Exception as e:
            logger.error(f"Error creating missing info section: {str(e)}")
            y -= spacing * 2
        
        # ======= Rejection Reasons (if rejected) ========
        try:
            if safe_get_value(candidate, "verdict", "").lower() == "reject":
                rejection_reasons = safe_get_list(candidate, "reasons_if_rejected")
                if rejection_reasons:
                    c.setFont("Helvetica-Bold", 12)
                    c.setFillColor(danger_color)
                    c.drawString(50, y, "❌ Rejection Reasons")
                    c.setFillColor(black)
                    y -= spacing
                    
                    c.setFont("Helvetica", 10)
                    for reason in rejection_reasons[:5]:  # Limit to 5 reasons
                        reason_lines = wrap_text(f"• {reason}", 85)
                        for line in reason_lines[:2]:  # Max 2 lines per reason
                            c.drawString(50, y, line)
                            y -= spacing * 0.7
                    
                    y -= spacing
        except Exception as e:
            logger.error(f"Error creating rejection reasons section: {str(e)}")
            y -= spacing * 2
        
        # ======= Recommendations ========
        try:
            recommendation = safe_get_value(candidate, "recommendation", "").strip()
            if recommendation and recommendation not in ["N/A", "n/a"]:
                c.setFont("Helvetica-Bold", 12)
                c.setFillColor(header_color)
                c.drawString(50, y, "🎯 Recommendations")
                c.setFillColor(black)
                y -= spacing
                
                c.setFont("Helvetica", 10)
                rec_lines = wrap_text(recommendation, 90)
                for line in rec_lines[:4]:  # Limit to 4 lines
                    c.drawString(50, y, line)
                    y -= spacing * 0.7
                
                y -= spacing
        except Exception as e:
            logger.error(f"Error creating recommendations section: {str(e)}")
            y -= spacing * 2
        
        # ======= Recruiter Notes (if any) ========
        try:
            notes = safe_get_value(candidate, "recruiter_notes", "").strip()
            if notes and notes not in ["N/A", "n/a"]:
                c.setFont("Helvetica-Bold", 12)
                c.setFillColor(header_color)
                c.drawString(50, y, "📝 Recruiter Notes")
                c.setFillColor(black)
                y -= spacing
                
                c.setFont("Helvetica", 10)
                notes_lines = wrap_text(notes, 90)
                for line in notes_lines[:6]:  # Limit to 6 lines
                    c.drawString(50, y, line)
                    y -= spacing * 0.7
                
                y -= spacing
        except Exception as e:
            logger.error(f"Error creating recruiter notes section: {str(e)}")
            y -= spacing * 2
        
        # ======= Footer ========
        try:
            c.setFont("Helvetica", 8)
            c.setFillColor(HexColor('#666666'))
            c.drawString(50, 50, f"Generated by EazyAI Resume Screener • Page 1")
            c.drawString(width - 200, 50, "Confidential Document")
            c.setFillColor(black)
        except Exception as e:
            logger.error(f"Error creating footer: {str(e)}")
        
        # Save the PDF
        c.save()
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error generating PDF summary: {str(e)}")
        # Return a basic error PDF
        try:
            buffer = BytesIO()
            c = canvas.Canvas(buffer, pagesize=A4)
            c.drawString(100, 750, "Error: Unable to generate candidate summary")
            c.drawString(100, 730, f"Error details: {str(e)}")
            c.drawString(100, 710, f"Candidate: {safe_get_value(candidate, 'name', 'Unknown')}")
            c.save()
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as final_error:
            logger.error(f"Failed to create error PDF: {str(final_error)}")
            # Return empty bytes if all else fails
            return b""

def create_batch_summary_pdf(candidates: List[Dict[str, Any]], job_title: str = "Position") -> bytes:
    """Create a batch summary PDF for multiple candidates"""
    try:
        buffer = BytesIO()
        c = canvas.Canvas(buffer, pagesize=A4)
        width, height = A4
        
        # Header
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, height - inch, f"Batch Candidate Summary - {job_title}")
        
        y = height - inch - 40
        spacing = 25
        
        for i, candidate in enumerate(candidates[:20]):  # Limit to 20 candidates per PDF
            if y < 100:  # Start new page
                c.showPage()
                y = height - inch
            
            name = safe_get_value(candidate, "name", f"Candidate {i+1}")
            score = safe_get_score(candidate, "score", 0)
            verdict = safe_get_value(candidate, "verdict", "review")
            
            # Color code by verdict
            if verdict == "shortlist":
                color = HexColor('#16a34a')
            elif verdict == "reject":
                color = HexColor('#dc2626')
            else:
                color = HexColor('#ea580c')
            
            c.setFont("Helvetica-Bold", 12)
            c.setFillColor(color)
            c.drawString(50, y, f"{i+1}. {name} - {score}% ({verdict.upper()})")
            c.setFillColor(black)
            
            # Add brief details
            c.setFont("Helvetica", 10)
            email = safe_get_value(candidate, "email", "N/A")
            c.drawString(70, y - 15, f"Email: {email}")
            
            fitment = safe_get_value(candidate, "fitment", "No analysis available")
            if len(fitment) > 80:
                fitment = fitment[:80] + "..."
            c.drawString(70, y - 30, f"Summary: {fitment}")
            
            y -= spacing * 2
        
        c.save()
        buffer.seek(0)
        return buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error creating batch PDF: {str(e)}")
        return b""
