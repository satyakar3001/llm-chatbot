from pathlib import Path
from shutil import copyfile
from langchain_community.document_loaders import TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.core.startup import initialize_models
import PyPDF2
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

embeddings, llm, vectordb, retriever, qa_chain = initialize_models()

def ingest_document(doc_name: str, doc_path: str):
    sops_dir = Path("sops")
    sops_dir.mkdir(exist_ok=True)
    dest_path = sops_dir / doc_name
    copyfile(doc_path, dest_path)
    documents = []
    if dest_path.suffix == ".txt":
        loader = TextLoader(str(dest_path), encoding="iso-8859-1")
        docs = loader.load()
    elif dest_path.suffix == ".docx":
        loader = Docx2txtLoader(str(dest_path))
        docs = loader.load()
    elif dest_path.suffix == ".pdf":
        pdf_text = ""
        with open(dest_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                pdf_text += page.extract_text() or ""
        if not pdf_text.strip():
            return False, "No text extracted from PDF."
        docs = [Document(page_content=pdf_text, metadata={"source": str(dest_path)})]
    else:
        return False, "Unsupported file type."
    documents.extend(docs)
    if not documents:
        return False, "No content loaded from document."
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    vectordb = Chroma.from_documents(
        chunks, embedding=embeddings, persist_directory=str(Path("chroma"))
    )
    vectordb.persist()
    return True, doc_name 