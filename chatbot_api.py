import os
import time
import yaml
from pathlib import Path
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings

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
# In-memory conversation history per session_id
session_histories = {}

class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"

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
        docs = response.get("source_documents", [])
        elapsed = round(time.time() - start_time, 2)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    # Update session history
    session_histories[session_id].append({"user": user_input, "bot": answer})
    # Remove sources from response
    return {
        "reply": answer,
        "elapsed": elapsed,
        "support_agent": support_agent,
        "session_id": session_id
    } 