from pydantic import BaseModel
from typing import List

class DeleteDocRequest(BaseModel):
    doc_name: str

class DeleteDocResponse(BaseModel):
    status: str
    deleted_file: str

class ListDocsResponse(BaseModel):
    documents: List[str] 