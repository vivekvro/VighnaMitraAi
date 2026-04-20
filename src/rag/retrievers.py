from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import Annotated,List
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda,RunnablePassthrough
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from src.rag.DocumentsLoader import DocLoader



def get_RetrievalQA(retriever):
    return RetrievalQA.from_chain_type(llm=ChatGroq(model="llama-3.3-70b-versatile"),retriever=retriever)




def get_retriever(
        doctype,upload_file,
        top_k:int=6,
        embeddings="BAAI/bge-base-en-v1.5"):
    if doctype not in ["pdf","txt","url"]:
        raise ValueError("Uploaded document is not valid.")
    loader = DocLoader(doctype=doctype,uploaded_file_path=upload_file)
    docs = loader.load()
    vec_store = FAISS.from_documents(documents=docs,embedding=HuggingFaceEmbeddings(model_name=embeddings))
    retriever = vec_store.as_retriever(search_type="mmr",search_kwargs={"k": top_k})
    return retriever


# user langchain.chains.retrieval_qa.base import RetrievalQA
#RetrievalQA.from_chain_type(llm=llm,retriever=retriever)