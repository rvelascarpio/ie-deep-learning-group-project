import base64
from io import BytesIO
from fpdf import FPDF
import datetime

class PDFReport(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 15)
        self.set_text_color(0, 188, 212) # Cyan
        self.cell(0, 10, 'FVJ Health-Tech | CardioAI(TM)', 0, 1, 'C')
        self.set_font('Helvetica', 'I', 10)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, 'Automated Neural Network Clinical Report', 0, 1, 'C')
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', 'I', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def safe_text(txt):
    return str(txt).encode('latin-1', 'replace').decode('latin-1')

def generate_pdf_report(patient_data, chart_bytes, predictions, thresholds, classes, verdict_title, verdict_text):
    pdf = PDFReport()
    pdf.add_page()
    
    # Patient Context
    pdf.set_font('Helvetica', 'B', 12)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 10, safe_text('Patient Demographic Context'), 0, 1)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, safe_text(f"Patient ID: {patient_data['patient_id']} | ECG ID: {patient_data['ecg_id']}"), 0, 1)
    pdf.cell(0, 6, safe_text(f"Age: {patient_data['age']} | Sex: {'Male' if patient_data['sex']==0 else 'Female'}"), 0, 1)
    pdf.cell(0, 6, safe_text(f"Report Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"), 0, 1)
    pdf.ln(5)

    # Diagnostic Verdict
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Model Diagnostic Verdict', 0, 1)
    pdf.set_font('Helvetica', 'B', 11)
    if 'ABNORMAL' in verdict_title:
        pdf.set_text_color(220, 38, 38) # Red
    elif 'NORMAL' in verdict_title:
        pdf.set_text_color(16, 185, 129) # Green
    else:
        pdf.set_text_color(245, 158, 11) # Orange
        
    pdf.cell(0, 8, safe_text(verdict_title.replace('⚠️', '').replace('✅', '').replace('ℹ️', '').strip()), 0, 1)
    pdf.set_font('Helvetica', '', 10)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(0, 6, safe_text(verdict_text))
    pdf.ln(5)

    # Probability Scores
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Multi-Label Probability Scores', 0, 1)
    pdf.set_font('Helvetica', '', 10)
    
    for c in classes:
        prob = predictions[classes.index(c)] * 100
        thresh = thresholds.get(c, 0.5) * 100
        status = "POSITIVE" if prob >= thresh else "NEGATIVE"
        if status == "POSITIVE":
            pdf.set_text_color(220, 38, 38)
        else:
            pdf.set_text_color(100, 100, 100)
        
        pdf.cell(0, 6, safe_text(f"{c}: {prob:.1f}% (Cutoff: {thresh:.1f}%) - {status}"), 0, 1)
        
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)

    # ECG Plot Image
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 10, 'Analyzed ECG Waveform', 0, 1)
    
    # Save chart_bytes to temp file since fpdf image() prefers a file path
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
        tmp.write(chart_bytes)
        tmp_path = tmp.name
        
    pdf.image(tmp_path, x=10, w=190)
    
    import os
    os.remove(tmp_path)

    return bytes(pdf.output(dest='S'))
