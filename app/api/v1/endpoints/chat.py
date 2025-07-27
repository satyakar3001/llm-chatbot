from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import chat_with_bot

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    answer, elapsed = chat_with_bot(request.message, request.session_id, {})
    return ChatResponse(
        reply=answer,
        elapsed=elapsed,
        session_id=request.session_id
    ) 