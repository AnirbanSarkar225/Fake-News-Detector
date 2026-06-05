from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from io import BytesIO
import datetime

def generate_credibility_pdf(title, prediction, confidence, text_snippet, summary, entities, sentiment, domain_profile=None):
    """
    Generate a styled PDF verification report using ReportLab and return its bytes.
    """
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom Color Palette matching the app (Organic Charcoal & Terracotta)
    c_primary = colors.HexColor("#090706") # Charcoal
    c_accent = colors.HexColor("#e15b3e") # Clay
    c_brass = colors.HexColor("#d49b4c") # Brass
    c_text = colors.HexColor("#333333")
    c_light_bg = colors.HexColor("#f9f7f2")
    
    # Define custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=24,
        textColor=c_primary,
        spaceAfter=15
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=c_accent,
        spaceBefore=12,
        spaceAfter=6,
        borderPadding=2
    )
    
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=10,
        textColor=c_text,
        leading=14,
        spaceAfter=8
    )
    
    badge_style = ParagraphStyle(
        'Badge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.white,
        alignment=1 # Center
    )
    
    # Document Header
    story.append(Paragraph("🛡️ TRUTHSHIELD VERIFICATION REPORT", title_style))
    story.append(Paragraph(f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", body_style))
    story.append(Spacer(1, 15))
    
    # Verdict Table (Skeuomorphic bezel representation)
    verdict_color = colors.HexColor("#5f8a6b") if prediction.upper() == "REAL" else colors.HexColor("#d45d4e")
    verdict_p = Paragraph(f"VERDICT: {prediction.upper()}", badge_style)
    conf_p = Paragraph(f"CONFIDENCE: {confidence*100:.2f}%", ParagraphStyle('Conf', parent=body_style, fontName='Helvetica-Bold', alignment=1))
    
    verdict_data = [
        [verdict_p],
        [conf_p]
    ]
    
    verdict_table = Table(verdict_data, colWidths=[200])
    verdict_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), verdict_color),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 10),
        ('LEFTPADDING', (0,0), (-1,-1), 15),
        ('RIGHTPADDING', (0,0), (-1,-1), 15),
        ('BACKGROUND', (0,1), (-1,1), c_light_bg),
        ('BOX', (0,0), (-1,-1), 1, c_primary),
    ]))
    
    # Align table to center
    container_table = Table([[verdict_table]], colWidths=[500])
    container_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER')
    ]))
    story.append(container_table)
    story.append(Spacer(1, 20))
    
    # Title analyzed
    story.append(Paragraph("<b>Article Title Analyzed:</b>", body_style))
    story.append(Paragraph(f'<i>"{title}"</i>', ParagraphStyle('ItalicTitle', parent=body_style, fontName='Helvetica-Oblique', fontSize=11, leading=15)))
    story.append(Spacer(1, 10))
    
    # Summary
    story.append(Paragraph("📖 Key Summarization", section_heading))
    story.append(Paragraph(summary, body_style))
    story.append(Spacer(1, 10))
    
    # Source Credibility Profile
    if domain_profile:
        story.append(Paragraph("🔗 Source Verification Profile", section_heading))
        source_text = f"<b>Domain:</b> {domain_profile['domain']}<br/>" \
                      f"<b>Trust Score:</b> {domain_profile['score']}% ({domain_profile['category']})<br/>" \
                      f"<b>Assessment:</b> {domain_profile['description']}"
        story.append(Paragraph(source_text, body_style))
        story.append(Spacer(1, 10))
        
    # Sentiment & Emotions
    story.append(Paragraph("🧠 Cognitive & Emotional Analysis", section_heading))
    sent_text = f"Fear score: {sentiment['fear']*100:.1f}% | " \
                f"Anger score: {sentiment['anger']*100:.1f}% | " \
                f"Neutral score: {sentiment['neutral']*100:.1f}%"
    story.append(Paragraph(sent_text, body_style))
    story.append(Spacer(1, 10))
    
    # Named Entities
    story.append(Paragraph("🏷️ Named Entity Profile", section_heading))
    entities_text = f"<b>Key People:</b> {', '.join(entities['people']) if entities['people'] else 'None identified'}<br/>" \
                    f"<b>Organizations:</b> {', '.join(entities['organizations']) if entities['organizations'] else 'None identified'}<br/>" \
                    f"<b>Locations:</b> {', '.join(entities['locations']) if entities['locations'] else 'None identified'}"
    story.append(Paragraph(entities_text, body_style))
    story.append(Spacer(1, 25))
    
    # Footer Disclaimer
    story.append(Paragraph("<font size=8 color='#777777'>Disclaimer: This analysis report is generated automatically based on machine learning classifications and NLP heuristics. It represents statistical probabilities of truthfulness rather than absolute validation. Always consult official and verified news agencies before drawing definitive conclusions.</font>", body_style))
    
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
