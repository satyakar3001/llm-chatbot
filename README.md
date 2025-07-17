# PDF Search Insights API

A FastAPI-based backend for extracting, searching, and summarizing relevant content from PDF documents using local NLP models.

## Features
- Register PDF documents for search
- Retrieve relevant information from PDFs using semantic search
- Summarize extracted content (paragraph, bullet points, or JSON)
- Uses only locally downloaded models (no internet required at runtime)

## Requirements
- Python 3.8+
- See `requirements.txt` for dependencies

## Setup

### 1. Clone the Repository
```bash
# Clone this repository and navigate to the project directory
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Required Models
Run the provided script to download the required models to the `models/` directory:
```bash
python download_models.py
```
This will download:
- SentenceTransformer: `paraphrase-MiniLM-L6-v2`
- HuggingFace Summarization: `facebook/bart-large-cnn`

### 4. Configure Environment Variables
Copy the template and edit if needed:
```bash
cp .envtemplate .env
```
Edit `.env` to set the paths to your local models if you use custom locations.

### 5. Run the API
```bash
uvicorn main:app --reload
```

## API Endpoints

### 1. Register a Document
`POST /submit_document`
```json
{
  "doc_name": "example_doc",
  "doc_path": "/path/to/your/document.pdf"
}
```

### 2. Retrieve Information
`POST /retrieve_information`
```json
{
  "doc_name": "example_doc",
  "query": "What is the summary of section 2?",
  "threshold": 0.1,
  "response_type": "paragraph" // or "bullet points" or "json"
}
```
**Returns:**
- `formatted_response`: The extracted and formatted text (with newlines removed)
- `page_number`: The page number where relevant content was found

## Notes
- All models are loaded from local disk only. No internet access is required at runtime.
- To add more models or change model paths, update your `.env` file.
- For production, use a process manager and disable `--reload`.

