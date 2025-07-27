import os
import sys
import traceback
from pathlib import Path

from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Config values from environment
DATA_DIR = os.getenv("DATA_DIR", None)
DB_DIR = os.getenv("DB_DIR", None)
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH", None)


def load_documents(folder):
    documents = []
    ingested_files = []

    for file_path in Path(folder).glob("*"):
        try:
            if file_path.suffix == ".txt":
                loader = TextLoader(str(file_path), encoding="iso-8859-1")
            elif file_path.suffix == ".docx":
                loader = Docx2txtLoader(str(file_path))
            else:
                print(f"Skipping unsupported file: {file_path.name}")
                continue
            docs = loader.load()
            documents.extend(docs)
            ingested_files.append(file_path.name)
            print(f"Loaded: {file_path.name}")
        except Exception as e:
            print(f"Failed to load {file_path.name}: {e}")
            traceback.print_exc()
    return documents, ingested_files


def ingest_docs_util(docs, ingested, embeddings):
    if not docs:
        print("No documents loaded. Check your files.")
        return False
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    vectordb = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=DB_DIR
    )
    vectordb.persist()
    print("Ingestion complete.")
    return True


def test_ingest_docs(vectordb, ingested):
    print("Files ingested:")
    for f in ingested:
        print(f" - {f}")
    retriever = vectordb.as_retriever()
    test_query = "How do I troubleshoot VPN connection issues?"
    results = retriever.get_relevant_documents(test_query)
    print("\nSample test query result:")
    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(doc.page_content[:500])


def ingest_docs():
    if not DATA_DIR or not DB_DIR or not EMBEDDING_MODEL_PATH:
        print("Please set the DATA_DIR, DB_DIR, and EMBEDDING_MODEL_PATH environment variables.")
        return False

    # Create embeddings instance once, reuse in ingestion and querying
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    docs, ingested = load_documents(DATA_DIR)
    if not ingest_docs_util(docs, ingested, embeddings):
        return False
    # For testing, load the vectordb and run a sample query:
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    test_ingest_docs(vectordb, ingested)
    return True


if __name__ == "__main__":
    ingest_docs()
