import os
import time
import yaml
import traceback
from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, Body
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from shutil import copyfile
import PyPDF2

# Load environment variables from .env
load_dotenv()

# === CONFIG ===
DATA_DIR = os.getenv("DATA_DIR", None)
DB_DIR = os.getenv("DB_DIR", None)
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", None)
LLM_MODEL_PATH = os.getenv("LLM_MODEL_PATH", None)
CONFIG_FILE = os.getenv("CONFIG_FILE", "config.yaml")
ATTEMPT_LIMIT = int(os.getenv("ATTEMPT_LIMIT", 2))

# --- LOAD SUPPORT AGENT INFO ---
def load_config():
    try:
        with open(CONFIG_FILE, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config.yaml: {e}")
        return {
            "support_agent": {
                "name": "Support Agent",
                "email": "support@example.com"
            }
        }

# --- FASTAPI APP ---
app = FastAPI()

# Allow CORS for all origins (customize as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBALS ---
embeddings = None
llm = None
vectordb = None
retriever = None
qa_chain = None
support_agent = None
session_histories = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

class IngestRequest(BaseModel):
    doc_name: str
    doc_path: str

class DeleteDocRequest(BaseModel):
    doc_name: str

@app.on_event("startup")
def startup_event():
    global embeddings, llm, vectordb, retriever, qa_chain, support_agent
    config = load_config()
    support_agent = config.get('support_agent', {})
    print(f"Support Agent: {support_agent.get('name')} <{support_agent.get('email')}>")
    print("Loading embedding model (offline)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    print("Loading local LLM (this may take a while)...")
    llm = LlamaCpp(
        model_path=LLM_MODEL_PATH,
        temperature=0.5,
        max_tokens=300,
        top_p=0.95,
        n_ctx=1024,
        verbose=False,
    )
    print("Loading vector DB...")
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    print("Setting up ConversationalRetrievalChain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=None,  # We'll handle memory per session
        return_source_documents=True,
        output_key="answer"
    )
    print("Chatbot API is ready!")

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    global qa_chain, support_agent, session_histories
    user_input = request.message.strip()
    session_id = request.session_id
    if session_id not in session_histories:
        session_histories[session_id] = []
    # Build memory for this session
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    # Replay history into memory
    for turn in session_histories[session_id]:
        memory.chat_memory.add_user_message(turn["user"])
        memory.chat_memory.add_ai_message(turn["bot"])
    # Create a new chain with session memory
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    start_time = time.time()
    try:
        response = chain({"question": user_input})
        answer = response.get("answer", "").strip()
        elapsed = round(time.time() - start_time, 2)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    # Update session history
    session_histories[session_id].append({"user": user_input, "bot": answer})
    return {
        "reply": answer,
        "elapsed": elapsed,
        "support_agent": support_agent,
        "session_id": session_id
    }

@app.post("/ingest")
async def ingest_endpoint(request: IngestRequest = Body(...)):
    global embeddings, vectordb
    try:
        doc_name = request.doc_name
        doc_path = request.doc_path
        sops_dir = Path("sops")
        sops_dir.mkdir(exist_ok=True)
        dest_path = sops_dir / doc_name
        copyfile(doc_path, dest_path)
        documents = []
        if dest_path.suffix == ".txt":
            loader = TextLoader(str(dest_path), encoding="iso-8859-1")
        elif dest_path.suffix == ".docx":
            loader = Docx2txtLoader(str(dest_path))
        elif dest_path.suffix == ".pdf":
            # PDF support using PyPDF2
            pdf_text = ""
            with open(dest_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    pdf_text += page.extract_text() or ""
            if not pdf_text.strip():
                return JSONResponse(status_code=400, content={"error": "No text extracted from PDF."})
            from langchain_core.documents import Document
            docs = [Document(page_content=pdf_text, metadata={"source": str(dest_path)})]
        else:
            return JSONResponse(status_code=400, content={"error": "Unsupported file type."})
        docs = loader.load()
        documents.extend(docs)
        if not documents:
            return JSONResponse(status_code=400, content={"error": "No content loaded from document."})
        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_documents(documents)
        vectordb = Chroma.from_documents(
            chunks, embedding=embeddings, persist_directory=DB_DIR
        )
        vectordb.persist()
        return {"status": "success", "ingested_file": doc_name, "count": 1}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/delete_doc")
async def delete_doc_endpoint(request: DeleteDocRequest = Body(...)):
    global vectordb
    try:
        doc_name = request.doc_name
        sops_dir = Path("sops")
        file_path = sops_dir / doc_name
        # Remove file from sops/
        if file_path.exists():
            file_path.unlink()
        else:
            return JSONResponse(status_code=404, content={"error": f"File {doc_name} not found in sops/ directory."})
        # Remove embedding from Chroma collection
        if vectordb is not None:
            # Chroma uses document IDs; we use the filename as the ID
            try:
                vectordb._collection.delete(where={"source": str(file_path)})
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": f"Failed to delete embedding: {e}"})
        return {"status": "success", "deleted_file": doc_name}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/list_docs")
async def list_docs_endpoint():
    sops_dir = Path("sops")
    if not sops_dir.exists():
        return {"documents": []}
    docs = [f.name for f in sops_dir.iterdir() if f.is_file()]
    return {"documents": docs} 