from fastapi import FastAPI,HTTPException
from src.rag.DocumentsLoader import DocLoader
from pydantic import BaseModel,Field
from typing import Annotated,Literal
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from src.rag.retrievers import update_vectorstore



from dotenv import load_dotenv
load_dotenv()

#-------------------------------------------------------------------------------

class FileDetails(BaseModel):
    path:Annotated[str,Field(description="Path to the document. Can be a URL or a local file path (e.g., from tempfile).")]
    doctype:Literal['pdf','txt','url']
    user_id:str

app =  FastAPI()


@app.post("/vm/upload_document")
def get_upload_docs(file:FileDetails):
    try:
        loader = DocLoader(doctype=file.doctype,path=file.path)
        docs = loader.load()
        if not docs:
            raise HTTPException(status_code=500,detail="NO document is loaded")
        if update_vectorstore(docs=docs,user_id=file.user_id):
            return {"response":"Uploaded Successfully"}
        else:
            return {"response":"something went wrong."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))