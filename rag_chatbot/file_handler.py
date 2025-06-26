import os
import pdfplumber
import docx

def extract_text_from_pdf(file_path):
    #Extracts text from a PDF file.
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=1)
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF file: {e}")
    return text

def extract_text_from_docx(file_path):
    #Extracts text from a DOCX file.
    text = ""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([para.text for para in doc.paragraphs])
    except Exception as e:
        print(f"Error reading DOCX file: {e}")
    return text

def get_text_from_file(file_path, ext):
    #Determines the file type and extracts text accordingly.
    if ext.lower() == ".pdf":
        return extract_text_from_pdf(file_path)
    elif ext.lower() in [".docx", ".doc"]:
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or DOCX file.")