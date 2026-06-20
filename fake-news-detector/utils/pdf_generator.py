"""
TruthShield PDF Report Generator
================================
Generates professional credibility verification reports as PDF documents.
Uses ReportLab for PDF creation with the project's color palette.

Falls back gracefully if ReportLab is not installed.
"""

import io
import os
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch, mm
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, KeepTogether, PageBreak
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

# ── Project color palette ──────────────────────────────────────────
ORGANIC_CHARCOAL = (9 / 255, 7 / 255, 6 / 255)           # #090706
CLAY            = (225 / 255, 91 / 255, 62 / 255)         # #e15b3e
BRASS           = (212 / 255, 155 / 255, 76 / 255)        # #d49b4c
SOFT_WHITE      = (245 / 255, 245 / 255, 240 / 255)       # #f5f5f0
DARK_BG         = (17 / 255, 24 / 255, 39 / 255)          # #111827
SLATE_BORDER    = (55 / 255, 65 / 255, 81 / 255)          # #374151

# Category colors (R, G, B floats)
CATEGORY_COLORS = {
    "REAL":                     (16 / 255, 185 / 255, 129 / 255),   # green  #10b981
    "HIGHLY CREDIBLE":          (16 / 255, 185 / 255, 129 / 255),   # green
    "LIKELY REAL":              (126 / 255, 163 / 255, 136 / 255),  # light green
    "UNCERTAIN":                (212 / 255, 155 / 255, 76 / 255),   # gold/brass
    "LIKELY FAKE":              (234 / 255, 88 / 255, 12 / 255),    # orange
    "HIGH RISK MISINFORMATION": (212 / 255, 93 / 255, 78 / 255),    # red
    "HIGH RISK":                (212 / 255, 93 / 255, 78 / 255),    # red
    "FAKE":                     (239 / 255, 68 / 255, 68 / 255),    # red    #ef4444
    "MISLEADING":               (249 / 255, 115 / 255, 22 / 255),   # orange #f97316
    "CLICKBAIT":                (245 / 255, 158 / 255, 11 / 255),   # yellow #f59e0b
    "SATIRE":                   (59 / 255, 130 / 255, 246 / 255),   # blue   #3b82f6
}

# ── Helpers ────────────────────────────────────────────────────────

def _rl_color(rgb_tuple):
    """Convert an (r, g, b) float tuple to a ReportLab Color."""
    return colors.Color(*rgb_tuple)


def _score_bar_text(value, width=20):
    """Return a simple ASCII-art bar like ████████░░░░░░░░░░░░ 40%."""
    filled = int(round(value / 100 * width))
    return "█" * filled + "░" * (width - filled) + f"  {value}%"


def _get_category_color(category):
    """Return the (r,g,b) tuple for a category string."""
    if not category:
        return SLATE_BORDER
    key = category.upper().strip()
    return CATEGORY_COLORS.get(key, SLATE_BORDER)


def _safe_str(value, default="N/A"):
    """Safely stringify a value."""
    if value is None:
        return default
    return str(value)


# ── Main generator ─────────────────────────────────────────────────

def generate_credibility_pdf(
    title,
    summary,
    prediction,
    confidence,
    credibility,
    indicators=None,
    domain_profile=None,
    sentiment=None,
    entities=None,
    category=None,
    clickbait_score=None,
    ai_score=None,
    verification_results=None,
    source_trust=None,
    explanation=None,
    reliability=None,
):
    """
    Generate a TruthShield Verification Report as a PDF.

    Parameters
    ----------
    title : str
        Article title / headline.
    summary : str
        Executive summary of the analysis.
    prediction : str
        Binary prediction label (e.g. "REAL" or "FAKE").
    confidence : float
        Model confidence score (0–1).
    credibility : float
        Overall credibility score (0–1).
    indicators : dict, optional
        Legacy indicator dict (e.g. {"emotional_language": 0.3}).
    domain_profile : dict, optional
        Source domain profile with keys like "domain", "category",
        "trust_score", "political_bias", "fact_check_record".
    sentiment : dict, optional
        Cognitive/emotional analysis with keys like "fear", "anger",
        "joy", "neutral", "surprise", "disgust", "sadness".
    entities : dict, optional
        Named entities with keys like "people", "organizations",
        "locations".
    category : str, optional
        Multi-class category label (REAL/FAKE/SATIRE/CLICKBAIT/MISLEADING).
    clickbait_score : float, optional
        Clickbait probability (0–1).
    ai_score : float, optional
        AI-generated content probability (0–1).
    verification_results : list[dict], optional
        Fact-check verification results, each with "claim"/"claim_text",
        "rating", and optionally "source".
    source_trust : float, optional
        Source trust percentage (0–100).
    explanation : str, optional
        Human-readable explanation for the classification.

    Returns
    -------
    bytes or None
        PDF content as bytes, or None if ReportLab is unavailable.
    """
    if not REPORTLAB_AVAILABLE:
        print("[TruthShield] ReportLab not installed — PDF generation unavailable.")
        print("[TruthShield] Install with: pip install reportlab")
        return None

    # Normalise inputs
    confidence_pct = round((confidence or 0) * 100)
    credibility_pct = round((credibility or 0) * 100)
    reliability_pct = round((reliability or 0) * 100) if reliability is not None else confidence_pct
    effective_category = category or prediction or "UNKNOWN"
    cat_color = _get_category_color(effective_category)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Compatibility mapping for domain profile keys
    if domain_profile and isinstance(domain_profile, dict):
        domain_profile = {k.lower(): v for k, v in domain_profile.items()}
        if 'score' in domain_profile and 'trust_score' not in domain_profile:
            domain_profile['trust_score'] = domain_profile['score']
        if 'bias' in domain_profile and 'political_bias' not in domain_profile:
            domain_profile['political_bias'] = domain_profile['bias']
        if 'description' in domain_profile and 'fact_check_record' not in domain_profile:
            domain_profile['fact_check_record'] = domain_profile['description']

    # ── Styles ──────────────────────────────────────────────────
    styles = getSampleStyleSheet()

    style_title = ParagraphStyle(
        "TSTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        textColor=_rl_color(CLAY),
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    style_subtitle = ParagraphStyle(
        "TSSubtitle",
        parent=styles["Normal"],
        fontSize=9,
        textColor=colors.grey,
        alignment=TA_CENTER,
        spaceAfter=12,
    )
    style_section = ParagraphStyle(
        "TSSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=_rl_color(BRASS),
        spaceBefore=16,
        spaceAfter=6,
    )
    style_body = ParagraphStyle(
        "TSBody",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.Color(0.15, 0.15, 0.15),
        leading=14,
        spaceAfter=6,
    )
    style_small = ParagraphStyle(
        "TSSmall",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.grey,
        leading=10,
        spaceAfter=4,
    )
    style_center = ParagraphStyle(
        "TSCenter",
        parent=style_body,
        alignment=TA_CENTER,
    )
    style_footer = ParagraphStyle(
        "TSFooter",
        parent=styles["Normal"],
        fontSize=7,
        textColor=colors.Color(0.5, 0.5, 0.5),
        alignment=TA_CENTER,
        spaceBefore=20,
    )

    # ── Build document ──────────────────────────────────────────
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        title="TruthShield Verification Report",
        author="TruthShield AI",
    )

    story = []

    # ── 1. Header ───────────────────────────────────────────────
    story.append(Paragraph("TRUTHSHIELD VERIFICATION REPORT", style_title))
    story.append(Paragraph(f"Generated: {now_str}", style_subtitle))
    story.append(HRFlowable(
        width="100%", thickness=1.5,
        color=_rl_color(CLAY), spaceAfter=14,
    ))

    # ── 2. Verdict table ────────────────────────────────────────
    verdict_data = [
        ["VERDICT", effective_category.upper()],
        ["Credibility Score", f"{credibility_pct}%"],
        ["Reliability Score", f"{reliability_pct}%"],
        ["Model Confidence", f"{confidence_pct}%"],
    ]
    verdict_table = Table(verdict_data, colWidths=[2.2 * inch, 3.8 * inch])
    verdict_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), _rl_color(DARK_BG)),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BACKGROUND", (1, 0), (1, 0), _rl_color(cat_color)),
        ("TEXTCOLOR", (1, 0), (1, 0), colors.white),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (1, 0), (1, 0), 14),
        ("GRID", (0, 0), (-1, -1), 0.5, _rl_color(SLATE_BORDER)),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(verdict_table)
    story.append(Spacer(1, 8))

    # ── 3. Score bars ───────────────────────────────────────────
    story.append(Paragraph(
        f"Credibility Score (Likely Truth):  {_score_bar_text(credibility_pct)}", style_body,
    ))
    story.append(Paragraph(
        f"Reliability Score (Evidence Density):   {_score_bar_text(reliability_pct)}", style_body,
    ))

    # ── 4. Category classification ──────────────────────────────
    story.append(Paragraph("Category Classification", style_section))
    story.append(Paragraph(
        f"This article is classified as <b>{effective_category.upper()}</b>.",
        style_body,
    ))
    if explanation:
        story.append(Paragraph(
            f"<i>Reason:</i> {explanation}", style_body,
        ))

    # ── 5. Article title ────────────────────────────────────────
    if title:
        story.append(Paragraph("Article Title", style_section))
        story.append(Paragraph(title, style_body))

    # ── 6. Executive summary ────────────────────────────────────
    if summary:
        story.append(Paragraph("Executive Summary", style_section))
        story.append(Paragraph(summary, style_body))

    # ── 7. Source verification profile ──────────────────────────
    if domain_profile and isinstance(domain_profile, dict):
        story.append(Paragraph("Source Verification Profile", style_section))
        dp_rows = [["Property", "Value"]]
        for key in ["domain", "category", "trust_score", "political_bias",
                     "fact_check_record", "country", "type"]:
            if key in domain_profile and domain_profile[key] is not None:
                label = key.replace("_", " ").title()
                dp_rows.append([label, _safe_str(domain_profile[key])])
        if len(dp_rows) > 1:
            dp_table = Table(dp_rows, colWidths=[2.2 * inch, 3.8 * inch])
            dp_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), _rl_color(DARK_BG)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, _rl_color(SLATE_BORDER)),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.97, 0.97, 0.95)]),
            ]))
            story.append(dp_table)

    # ── 8. Clickbait analysis ───────────────────────────────────
    if clickbait_score is not None:
        cb_pct = round(clickbait_score * 100)
        story.append(Paragraph("Clickbait Analysis", style_section))
        story.append(Paragraph(
            f"Clickbait Probability: <b>{cb_pct}%</b>", style_body,
        ))
        story.append(Paragraph(
            _score_bar_text(cb_pct), style_small,
        ))
        if cb_pct >= 70:
            story.append(Paragraph(
                "<font color='#e15b3e'>⚠ High clickbait probability detected. "
                "The headline may be sensationalised.</font>", style_body,
            ))

    # ── 9. AI-generated content probability ─────────────────────
    if ai_score is not None:
        ai_pct = round(ai_score * 100)
        story.append(Paragraph("AI-Generated Content Analysis", style_section))
        story.append(Paragraph(
            f"AI-Generated Probability: <b>{ai_pct}%</b>", style_body,
        ))
        story.append(Paragraph(
            _score_bar_text(ai_pct), style_small,
        ))
        if ai_pct >= 60:
            story.append(Paragraph(
                "<font color='#e15b3e'>⚠ This content shows significant markers "
                "of AI-generated text.</font>", style_body,
            ))

    # ── 10. Fact verification results ───────────────────────────
    if verification_results and isinstance(verification_results, list) and len(verification_results) > 0:
        story.append(Paragraph("Fact Verification Results", style_section))
        vr_rows = [["#", "Claim", "Rating", "Source"]]
        for idx, vr in enumerate(verification_results, start=1):
            claim = vr.get("claim_text") or vr.get("claim") or "—"
            rating = vr.get("rating") or "—"
            source = vr.get("source") or "—"
            # Truncate long claims
            if len(claim) > 80:
                claim = claim[:77] + "..."
            vr_rows.append([str(idx), claim, rating, source])

        vr_table = Table(vr_rows, colWidths=[0.4 * inch, 3 * inch, 1.2 * inch, 1.4 * inch])
        vr_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), _rl_color(DARK_BG)),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, _rl_color(SLATE_BORDER)),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.97, 0.97, 0.95)]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]))
        story.append(vr_table)

    # ── 11. Cognitive & emotional analysis ──────────────────────
    if sentiment and isinstance(sentiment, dict):
        story.append(Paragraph("Cognitive &amp; Emotional Analysis", style_section))
        emotion_keys = ["fear", "anger", "joy", "neutral", "surprise", "disgust", "sadness"]
        em_rows = [["Emotion", "Score", "Visual"]]
        for emo in emotion_keys:
            if emo in sentiment and sentiment[emo] is not None:
                score = sentiment[emo]
                pct = round(score * 100) if isinstance(score, float) and score <= 1 else round(score)
                em_rows.append([
                    emo.capitalize(),
                    f"{pct}%",
                    _score_bar_text(pct, width=15),
                ])
        if len(em_rows) > 1:
            em_table = Table(em_rows, colWidths=[1.2 * inch, 0.8 * inch, 4 * inch])
            em_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), _rl_color(DARK_BG)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, _rl_color(SLATE_BORDER)),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.97, 0.97, 0.95)]),
            ]))
            story.append(em_table)

    # ── 12. Named entity profile ────────────────────────────────
    if entities and isinstance(entities, dict):
        story.append(Paragraph("Named Entity Profile", style_section))
        entity_types = {
            "people": "👤 People",
            "persons": "👤 People",
            "organizations": "🏢 Organizations",
            "orgs": "🏢 Organizations",
            "locations": "📍 Locations",
            "gpe": "📍 Locations",
        }
        for key, label in entity_types.items():
            if key in entities and entities[key]:
                ent_list = entities[key]
                if isinstance(ent_list, list):
                    ent_str = ", ".join(str(e) for e in ent_list[:15])
                else:
                    ent_str = str(ent_list)
                story.append(Paragraph(
                    f"<b>{label}:</b> {ent_str}", style_body,
                ))

    # ── 13. Explainability ──────────────────────────────────────
    if explanation:
        story.append(Paragraph("Explainability — Classification Rationale", style_section))
        story.append(Paragraph(explanation, style_body))

    # ── Source trust (standalone) ───────────────────────────────
    if source_trust is not None:
        st_pct = round(source_trust)
        story.append(Paragraph("Source Trust Score", style_section))
        story.append(Paragraph(
            f"Overall Source Trust: <b>{st_pct}%</b>", style_body,
        ))
        story.append(Paragraph(
            _score_bar_text(st_pct), style_small,
        ))

    # ── Legacy indicators (backward compat) ────────────────────
    if indicators and isinstance(indicators, dict):
        story.append(Paragraph("Analysis Indicators", style_section))
        ind_rows = [["Indicator", "Score"]]
        for ind_key, ind_val in indicators.items():
            label = ind_key.replace("_", " ").title()
            if isinstance(ind_val, float):
                display = f"{round(ind_val * 100)}%"
            else:
                display = _safe_str(ind_val)
            ind_rows.append([label, display])
        if len(ind_rows) > 1:
            ind_table = Table(ind_rows, colWidths=[3 * inch, 3 * inch])
            ind_table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), _rl_color(DARK_BG)),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.5, _rl_color(SLATE_BORDER)),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.Color(0.97, 0.97, 0.95)]),
            ]))
            story.append(ind_table)

    # ── 14. Footer disclaimer ───────────────────────────────────
    story.append(Spacer(1, 24))
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.Color(0.7, 0.7, 0.7), spaceAfter=6,
    ))
    story.append(Paragraph(
        "DISCLAIMER: This report is generated by TruthShield AI and is intended "
        "for informational purposes only. It does not constitute professional "
        "fact-checking or journalistic verification. Always cross-reference "
        "claims with authoritative sources. AI-based analysis may contain "
        "errors or biases.",
        style_footer,
    ))
    story.append(Paragraph(
        f"TruthShield Verification Report • {now_str} • Confidential",
        style_footer,
    ))

    # ── Build PDF ───────────────────────────────────────────────
    try:
        doc.build(story)
    except Exception as e:
        print(f"[TruthShield] Error building PDF: {e}")
        return None

    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes


def save_credibility_pdf(filepath, **kwargs):
    """
    Generate and save a credibility PDF to disk.

    Parameters
    ----------
    filepath : str
        Output path for the PDF file.
    **kwargs
        All keyword arguments are passed to ``generate_credibility_pdf``.

    Returns
    -------
    str or None
        The filepath on success, None on failure.
    """
    pdf_bytes = generate_credibility_pdf(**kwargs)
    if pdf_bytes is None:
        return None

    try:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "wb") as f:
            f.write(pdf_bytes)
        return filepath
    except Exception as e:
        print(f"[TruthShield] Error saving PDF to {filepath}: {e}")
        return None
