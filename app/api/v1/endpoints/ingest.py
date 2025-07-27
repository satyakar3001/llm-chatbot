from fastapi import APIRouter, Body
from app.schemas.ingest import IngestRequest, IngestResponse
from app.services.ingest_service import ingest_document

router = APIRouter()

@router.post("/ingest", response_model=IngestResponse)
async def ingest_endpoint(request: IngestRequest = Body(...)):
    success, result = ingest_document(request.doc_name, request.doc_path)
    if not success:
        return {"status": "error", "ingested_file": result, "count": 0}
    return IngestResponse(status="success", ingested_file=result, count=1) 