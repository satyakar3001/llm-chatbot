# Conversational Document-Aware Chatbot (Modular FastAPI Edition)

This project is a modular, production-ready FastAPI chatbot that answers questions based on your documents. It supports persistent chat session history in PostgreSQL (viewable in pgAdmin), document ingestion, and vector search using ChromaDB.

---

## Features
- **Modular FastAPI structure** for maintainability and scalability
- **Document ingestion** (.txt, .docx, .pdf) and vector embedding
- **Conversational AI** with context-aware responses
- **Session history persisted in PostgreSQL** (viewable in pgAdmin)
- **ChromaDB** for semantic search
- **Easy to extend** with new endpoints, models, or services

---

## Directory Structure
```
llm-chatbot-new/
├── app/
│   ├── api/v1/endpoints/      # All route definitions
│   ├── core/                  # Config, startup
│   ├── db/                    # DB session, base
│   ├── models/                # SQLAlchemy models
│   ├── schemas/               # Pydantic schemas
│   ├── services/              # Business logic
│   └── main.py                # FastAPI entry point
├── chroma/                    # ChromaDB persistence
├── sops/                      # Ingested documents
├── .envtemplate               # Environment variable template
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── ...                        # (other legacy files/folders)
```

---

## Environment Variables (.env)
Copy `.envtemplate` to `.env` and fill in your values:
```
DATABASE_URL=postgresql://username:password@localhost:5432/yourdb
DATA_DIR=./data
DB_DIR=./chroma
EMBEDDING_MODEL_PATH=./models/all-MiniLM-L6-v2
LLM_MODEL_PATH=./models/llama-2-7b-chat
```

---

## Database Setup
1. **Start PostgreSQL** and create your database (e.g., `conversation_chatbot`).
2. **Create the chat_sessions table** (in pgAdmin or psql):
   ```sql
   CREATE TABLE chat_sessions (
       id SERIAL PRIMARY KEY,
       session_id VARCHAR(255) NOT NULL,
       user_message TEXT NOT NULL,
       bot_message TEXT NOT NULL,
       timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
   );
   ```
3. Or, run the SQLAlchemy table creation script (see project instructions).

---

## Installation & Running
1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the app:**
   ```bash
   uvicorn app.main:app --reload
   ```
3. **Access the API docs:**
   - [http://localhost:8000/docs](http://localhost:8000/docs)

---

## API Endpoints
- `POST /api/v1/chat` — Chat with the bot (session-aware)
- `POST /api/v1/ingest` — Ingest a document
- `POST /api/v1/delete_doc` — Delete a document and its embedding
- `GET  /api/v1/list_docs` — List all ingested documents

---

## Notes
- **Session history is now persistent in PostgreSQL.**
- **ChromaDB** stores vector embeddings in the `chroma/` directory.
- **Ingested documents** are stored in the `sops/` directory.
- **Legacy files** (`main_api.py`, `chatbot_api.py`, `conv_chatbot.py`, `ingest_docs.py`) are no longer used.

---

## Extending
- Add new endpoints in `app/api/v1/endpoints/`
- Add new business logic in `app/services/`
- Add new models in `app/models/`

---

## License
MIT (or specify your own) 