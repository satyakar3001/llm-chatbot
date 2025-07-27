from langchain.memory import ConversationBufferMemory
from app.core.startup import initialize_models

embeddings, llm, vectordb, retriever, qa_chain = initialize_models()


def chat_with_bot(user_input: str, session_id: str, support_agent: dict):
    from app.models.session import ChatSession
    from app.db.session import SessionLocal
    import time
    start_time = time.time()

    def save_chat_turn(session_id, user, bot):
        db = SessionLocal()
        db.add(ChatSession(session_id=session_id, user_message=user, bot_message=bot))
        db.commit()
        db.close()

    def load_chat_history(session_id):
        db = SessionLocal()
        turns = db.query(ChatSession).filter_by(session_id=session_id).order_by(ChatSession.timestamp).all()
        db.close()
        return [{"user": t.user_message, "bot": t.bot_message} for t in turns]

    # if session_id not in load_chat_history(session_id):
    #     save_chat_turn(session_id, "", "") # Initialize session if it doesn't exist

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    for turn in load_chat_history(session_id):
        memory.chat_memory.add_user_message(turn["user"])
        memory.chat_memory.add_ai_message(turn["bot"])
    from langchain.chains import ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    response = chain({"question": user_input})
    answer = response.get("answer", "").strip()
    elapsed = round(time.time() - start_time, 2)
    save_chat_turn(session_id, user_input, answer)
    return answer, elapsed 