from pathlib import Path
from app.core.startup import initialize_models

embeddings, llm, vectordb, retriever, qa_chain = initialize_models()

def delete_document(doc_name: str):
    sops_dir = Path("sops")
    file_path = sops_dir / doc_name
    if file_path.exists():
        file_path.unlink()
    else:
        return False, f"File {doc_name} not found in sops/ directory."
    if vectordb is not None:
        try:
            vectordb._collection.delete(where={"source": str(file_path)})
        except Exception as e:
            return False, f"Failed to delete embedding: {e}"
    return True, doc_name

def list_documents():
    sops_dir = Path("sops")
    if not sops_dir.exists():
        return []
    return [f.name for f in sops_dir.iterdir() if f.is_file()] 