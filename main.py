# app.py

import os
import re
import base64
import io
import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
from sentence_transformers import SentenceTransformer, util
from transformers.pipelines import pipeline
from io import BytesIO
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
SENTENCE_TRANSFORMER_MODEL = os.environ.get('SENTENCE_TRANSFORMER_MODEL', './models/paraphrase-MiniLM-L6-v2')
SUMMARIZATION_MODEL = os.environ.get('SUMMARIZATION_MODEL', './models/bart-large-cnn')

app = FastAPI()
documents = {}

class SubmitDocumentRequest(BaseModel):
    doc_name: str
    doc_path: str

class RetrieveInformationRequest(BaseModel):
    doc_name: str
    query: str
    threshold: float = 0.1
    response_type: str  # "paragraph", "bullet points"

def check_page_for_consecutive_non_sentence_lines(page):
    text = page.get_text()
    lines = text.splitlines()
    consecutive_non_sentence_count = 0
    for line in lines:
        if line.strip() and any(char.isalpha() for char in line):
            if len(line.split()) <= 3:
                consecutive_non_sentence_count += 1
            else:
                consecutive_non_sentence_count = 0
        else:
            consecutive_non_sentence_count += 1
        if consecutive_non_sentence_count >= 5:
            return True
    return False

def extract_relevant_content(pdf_path, prompt, model_name=None, similarity_threshold=0.1):
    model_path = model_name if model_name else SENTENCE_TRANSFORMER_MODEL
    model = SentenceTransformer(model_path)
    doc = fitz.open(pdf_path)
    relevant_text = []
    prompt_embedding = model.encode(prompt, convert_to_tensor=True)
    page_num_final = None
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()  # type: ignore
        if text.strip():
            text_embedding = model.encode(text, convert_to_tensor=True)
            similarity_score = util.cos_sim(prompt_embedding, text_embedding).item()
            print(f"Page {page_num+1} similarity score: {similarity_score}")
            if similarity_score > similarity_threshold:
                relevant_text.append(text.strip())
                page_num_final = page_num + 1
    combined_text = "\n\n".join(relevant_text)
    return {
        "relevant_text": combined_text,
        "page_number": page_num_final
    }

def summarize_text(text):
    summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL, tokenizer=SUMMARIZATION_MODEL, local_files_only=True)
    summary_result = summarizer(text, max_length=130, min_length=30, do_sample=False)
    if isinstance(summary_result, list) and len(summary_result) > 0 and 'summary_text' in summary_result[0]:
        return summary_result[0]['summary_text']
    return ""

def format_output(text, format_prompt):
    if not text:
        return text
    if "summarize" in format_prompt.lower():
        return summarize_text(text)
    sentences = re.split(r"(?<=[.!?]) +", text.strip())
    if "bullet points" in format_prompt.lower():
        formatted_text = "\n".join(f"- {sentence.strip()}" for sentence in sentences if sentence.strip())
        return formatted_text
    elif "paragraph" in format_prompt.lower():
        return text
    elif "json" in format_prompt.lower():
        json_format = {f"sentence_{i+1}": sentence.strip() for i, sentence in enumerate(sentences) if sentence.strip()}
        return json.dumps(json_format, indent=2)
    return text

@app.post("/submit_document")
def submit_document(request: SubmitDocumentRequest):
    if not os.path.exists(request.doc_path):
        raise HTTPException(status_code=404, detail="Document path not found")
    documents[request.doc_name] = request.doc_path
    return {"message": f"Document '{request.doc_name}' registered successfully."}

@app.post("/retrieve_information")
def retrieve_information(request: RetrieveInformationRequest):
    if request.doc_name not in documents:
        raise HTTPException(status_code=404, detail="Document not found")
    pdf_path = documents[request.doc_name]
    result = extract_relevant_content(pdf_path, request.query, SENTENCE_TRANSFORMER_MODEL, similarity_threshold=request.threshold)
    if not result["relevant_text"]:
        return {"message": "No relevant text found for this query."}
    formatted = format_output(result["relevant_text"], request.response_type)
    formatted = formatted.replace("\n", "") if isinstance(formatted, str) else formatted
    return {
        "formatted_response": formatted,
        "page_number": result["page_number"]
    }
