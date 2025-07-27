# Conversational Document-Aware Chatbot

This project is a Python-based conversational bot that can understand user queries and provide context-aware answers based on the documents you provide. It leverages modern LLMs, semantic search, and vector databases to deliver accurate, document-grounded responses.

## Features
- **Conversational AI:** Maintains context across turns for natural dialogue.
- **Document Ingestion:** Supports ingestion of `.docx` and `.txt` files.
- **Semantic Search:** Uses HuggingFace embeddings and ChromaDB for fast, relevant retrieval.
- **Local LLM Support:** Can run with a local LLM (e.g., Llama) for offline or private deployments.
- **Extensible:** Built with LangChain for easy extension to new file types, models, or vector stores.

## Project Structure
```
llm-chatbot-new/
├── ingest_docs.py        # Script to ingest and embed documents
├── conv_chatbot.py       # Main conversational chatbot script
├── models/               # (gitignored) LLM and embedding models
├── sops/                 # (gitignored) Standard operating procedures or sensitive docs
├── chroma/               # (gitignored) ChromaDB persistence directory
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd llm-chatbot-new
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv env
# On Windows:
env\Scripts\activate
# On Linux/Mac:
source env/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*If `requirements.txt` is missing, install manually:*
```bash
pip install langchain langchain-community chromadb sentence-transformers torch python-docx
```

### 4. Download or Place Your LLM and Embedding Models
- Place your local LLM and embedding models in the `models/` directory (see your `conv_chatbot.py` and `ingest_docs.py` for model path variables).

### 5. Ingest Documents
Place your `.docx` or `.txt` files in the data directory (as configured in your scripts), then run:
```bash
python ingest_docs.py
```

### 6. Start the Conversational Bot
```bash
python conv_chatbot.py
```

## Usage
- Interact with the bot in your terminal.
- Type your questions; the bot will answer using the ingested documents.
- Type `exit` to quit the chatbot.

## Troubleshooting & Notes
- **Deprecation Warnings:**
  - `HuggingFaceEmbeddings` and `Chroma` are deprecated in recent LangChain versions. For long-term use, migrate to `langchain-huggingface` and `langchain-chroma` packages.
  - See [LangChain migration guide](https://python.langchain.com/docs/versions/migrating_memory/) for updating memory usage.
- **Model Loading:**
  - Ensure your model paths are correct and models are compatible with your hardware (CPU/GPU).
- **Document Support:**
  - Only `.docx` and `.txt` files are supported by default. Extend `ingest_docs.py` to add more types.
- **Vector DB Persistence:**
  - The `chroma/` directory stores your vector database and is gitignored by default.

## License
MIT License (or specify your own)

## Acknowledgements
- [LangChain](https://github.com/langchain-ai/langchain)
- [ChromaDB](https://www.trychroma.com/)
- [HuggingFace Transformers](https://huggingface.co/)
- [Llama.cpp](https://github.com/ggerganov/llama.cpp) (if using local LLM) 