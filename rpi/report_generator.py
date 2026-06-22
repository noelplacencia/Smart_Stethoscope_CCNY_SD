from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet
import os


def generate_report(patient_data, filename="reports/report.pdf"):
    os.makedirs("reports", exist_ok=True)

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph("Smart Stethoscope Clinical Report", styles["Title"])
    )

    content.append(Spacer(1, 12))

    for key, value in patient_data.items():
        content.append(
            Paragraph(f"<b>{key}</b>: {value}", styles["Normal"])
        )

    doc.build(content)

    return filename
