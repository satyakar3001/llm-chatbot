from pydantic import BaseModel

class IngestRequest(BaseModel):
    doc_name: str
    doc_path: str

class IngestResponse(BaseModel):
    status: str
    ingested_file: str
    count: int 