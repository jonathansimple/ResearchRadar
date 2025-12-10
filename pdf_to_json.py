import os
import re
import json
import PyPDF2

def extract_text_from_pdf(pdf_path):
    """Extract all text from a PDF file."""
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            if page.extract_text():
                text += page.extract_text() + "\n"
    return text

def split_into_paragraphs(text):
    """
    Split text into paragraphs using single newlines.
    Heuristic: 
      - If a line ends with punctuation (.!?), treat it as end of paragraph.
      - Otherwise, join it with the next line.
    """
    text = text.replace("\r", "\n")
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    paragraphs = []
    buffer = []

    for line in lines:
        buffer.append(line)
        if re.search(r"[。.!?]$", line):  # line ends with sentence-ending punctuation
            paragraphs.append(" ".join(buffer))
            buffer = []

    if buffer:  # leftover lines
        paragraphs.append(" ".join(buffer))

    return paragraphs

def pdf_to_json(pdf_path):
    text = extract_text_from_pdf(pdf_path)
    paragraphs = split_into_paragraphs(text)
    data = [{"idx": i, "text": p} for i, p in enumerate(paragraphs)]
    return data

def process_directory(pdf_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(pdf_dir):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_dir, filename)
            json_data = pdf_to_json(pdf_path)

            output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_corpus.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            print(f"Processed {filename} → {output_file}")

if __name__ == "__main__":
    input_dir = "recent_arxiv_cs_AI"       # directory containing PDF files
    output_dir = "recent_arxiv_cs_AI_json"  # directory to save JSON results
    process_directory(input_dir, output_dir)

