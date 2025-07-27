import os
import sys
import time
import yaml
import traceback
from pathlib import Path

# from langchain_community.document_loaders import TextLoader, UnstructuredDocxLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import load_dotenv

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


# --- MODEL LOADING ---
def load_models():
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
    return embeddings, llm

# --- CHATBOT LOGIC ---
def get_response(qa_chain, support_agent, session_id=None):
    chat_history = []
    failed_attempts = 0
    chatCounter = 0
    while True:
        Flag = True
        user_input = input(f"\n{support_agent.get('name')}: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Thanks for chatting with me! Goodbye!")
            break
        start_time = time.time()
        try:
            response = qa_chain({"question": user_input})
            answer = response.get("answer", "").strip()
            docs = response.get("source_documents", [])
            elapsed = round(time.time() - start_time, 2)
        except Exception as e:
            print(f"\nError during chat: {e}")
            continue

        if not answer or not docs or "couldn't find" in answer.lower():
            print(f"\nConversational Agent (attempt {failed_attempts+1}): {answer or 'Sorry, no info found.'} [{elapsed}s]")
            Flag = False
        else:
            print(f"\nConversational Agent (step {chatCounter+1}): {answer} [{elapsed}s]")

        chat_history.append(f"{support_agent.get('name')}: {user_input}\nConversational Agent: {answer}")

        resolved = input("Did that fix the issue? (yes/no): ").strip().lower()
        if resolved in ["yes", "y"]:
            print("Glad I could help!")
            break
        elif resolved in ["no", "n"]:
            Flag = False

        if not Flag:
            failed_attempts += 1
        else:
            failed_attempts = 0
            chatCounter += 1

        if failed_attempts >= ATTEMPT_LIMIT:
            print("\nInvoking escalation to support agent.")
            print(f"Contact: {support_agent.get('name')} <{support_agent.get('email')}>")
            print("Chat Summary:")
            for line in chat_history:
                print(line)
            break

def chatbot_agent():
    config = load_config()
    support_agent = config.get('support_agent', {})
    print(f"Support Agent: {support_agent.get('name')} <{support_agent.get('email')}>")
    # --- Load model (offline) ---
    embeddings, llm = load_models()
    # --- Load vector store ---
    print("Loading vector DB...")
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    # --- Memory + Conversational Retrieval Chain ---
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        ),
        return_source_documents=True,
        output_key="answer"
    )
    # --- Chat Loop ---
    print("*" * 60)
    print("Chatbot is ready! Type 'exit' to quit.")
    get_response(qa_chain, support_agent)

if __name__ == "__main__":
    chatbot_agent()
