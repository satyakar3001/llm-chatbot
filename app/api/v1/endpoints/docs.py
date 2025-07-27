from fastapi import APIRouter, Body
from app.schemas.docs import DeleteDocRequest, DeleteDocResponse, ListDocsResponse
from app.services.docs_service import delete_document, list_documents

router = APIRouter()

@router.post("/delete_doc", response_model=DeleteDocResponse)
async def delete_doc_endpoint(request: DeleteDocRequest = Body(...)):
    success, result = delete_document(request.doc_name)
    if not success:
        return {"status": "error", "deleted_file": result}
    return DeleteDocResponse(status="success", deleted_file=result)

@router.get("/list_docs", response_model=ListDocsResponse)
async def list_docs_endpoint():
    docs = list_documents()
    return ListDocsResponse(documents=docs) 