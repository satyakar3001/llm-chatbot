from langchain_community.vectorstores import Chroma
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFaceEmbeddings
from .config import DATA_DIR, DB_DIR, EMBEDDING_MODEL_PATH, LLM_MODEL_PATH

def initialize_models():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_PATH,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    llm = LlamaCpp(
        model_path=LLM_MODEL_PATH,
        temperature=0.5,
        max_tokens=300,
        top_p=0.95,
        n_ctx=1024,
        verbose=False,
    )
    vectordb = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=None,
        return_source_documents=True,
        output_key="answer"
    )
    return embeddings, llm, vectordb, retriever, qa_chain 