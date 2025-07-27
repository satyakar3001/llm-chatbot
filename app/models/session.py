from sqlalchemy import Column, Integer, String, Text, TIMESTAMP
from app.db.base import Base

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(255), index=True)
    user_message = Column(Text, nullable=False)
    bot_message = Column(Text, nullable=False)
    timestamp = Column(TIMESTAMP, server_default="now()")