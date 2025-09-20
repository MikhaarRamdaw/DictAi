# utils/export.py
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def export_pdf(soap_dict):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    text = c.beginText(50, height - 50)
    text.setFont("Helvetica", 12)

    for section, lines in soap_dict.items():
        text.textLine(f"{section.upper()}:")
        for ln in lines:
            text.textLine(f" - {ln}")
        text.textLine("")

    c.drawText(text)
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()
