# utils/render.py

from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.enums import TA_LEFT

# -----------------------------
# Markdown Rendering (for Streamlit display)
# -----------------------------
def render_markdown(grouped: dict) -> str:
    """Convert grouped SOAP dict into Markdown string."""
    md = "## SOAP Note\n\n"
    for section, sentences in grouped.items():
        md += f"### {section.capitalize()}\n"
        if sentences:
            for s in sentences:
                md += f"- {s}\n"
        else:
            md += "-\n"
        md += "\n"
    return md


# -----------------------------
# PDF Rendering
# -----------------------------
def render_pdf(grouped: dict) -> bytes:
    """Convert grouped SOAP dict into PDF bytes."""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)

    styles = getSampleStyleSheet()
    normal = styles["Normal"]
    normal.fontSize = 11
    normal.leading = 14

    section_style = ParagraphStyle(
        "Section",
        parent=styles["Heading2"],
        alignment=TA_LEFT,
        fontSize=14,
        spaceAfter=6,
    )

    story = []
    story.append(Paragraph("SOAP Note", styles["Heading1"]))
    story.append(Spacer(1, 12))

    for section, sentences in grouped.items():
        story.append(Paragraph(section.capitalize(), section_style))
        if sentences:
            for s in sentences:
                story.append(Paragraph(f"• {s}", normal))
        else:
            story.append(Paragraph("•", normal))
        story.append(Spacer(1, 8))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
