from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from typing import Annotated,List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from src.rag.DocumentsLoader import DocLoader
from pathlib import Path


embedding = HuggingFaceBgeEmbeddings()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
VECTORSTORE_DIR_PATH = BASE_DIR / "data" / "vectorstore"


def get_vectorstore_path(user_id: str):
    return VECTORSTORE_DIR_PATH / user_id


def create_vectorstore(user_id: str, docs):
    path = get_vectorstore_path(user_id)
    path.mkdir(parents=True, exist_ok=True)

    vectorstore = FAISS.from_documents(docs, embedding=embedding)
    vectorstore.save_local(str(path))
    return vectorstore


def load_vectorstore(user_id: str):
    path = get_vectorstore_path(user_id)

    if not path.exists():
        return None

    return FAISS.load_local(
        str(path),
        embedding,
        allow_dangerous_deserialization=True
    )


def update_vectorstore(docs, user_id: str):
    path = get_vectorstore_path(user_id)
    vectorstore = load_vectorstore(user_id)

    if vectorstore is None:
        vectorstore = create_vectorstore(user_id, docs)
    else:
        vectorstore.add_documents(docs)
        vectorstore.save_local(str(path))

    return vectorstore





def get_RetrievalQA(retriever):
    return RetrievalQA.from_chain_type(llm=ChatGroq(model="llama-3.3-70b-versatile"),retriever=retriever)



# user langchain.chains.retrieval_qa.base import RetrievalQA
#RetrievalQA.from_chain_type(llm=llm,retriever=retriever)



if __name__=="__main__":
    print(str(VECTORSTORE_DIR_PATH))