import os
import pymupdf4llm
import docx

def extract_text_from_pdf(file_path):
    #Extracts text from a PDF file.
    text = ""
    try:
        with open(file_path, 'r') as pdf:
            text = pymupdf4llm.to_markdown(pdf)
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