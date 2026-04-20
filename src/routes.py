from fastapi import FastAPI,HTTPException
from src.rag.DocumentsLoader import DocLoader
from pydantic import BaseModel,Field
from typing import Annotated,Literal


class FileDetails(BaseModel):
    path:Annotated[str,Field(description="Path to the document. Can be a URL or a local file path (e.g., from tempfile).")]
    doctype:Literal['pdf','txt','url']

app =  FastAPI()

@app.post("vm/upload_document")
def get_upload_docs(file:FileDetails):
    try:
        loader = DocLoader(doctype=file.doctype,path=file.path)
        docs = loader.load()
        return {"response":docs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



