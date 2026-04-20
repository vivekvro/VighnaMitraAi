import tempfile
from typing import Annotated,Literal
from langchain_community.document_loaders import WebBaseLoader,PyMuPDFLoader,TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter



def load_tempfile_path(upload_file):
    if upload_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmpfile:
            tmpfile.write(upload_file.read())
            return tmpfile.name


class DocLoader:
    def __init__(
            self,
            doctype:Literal["pdf","txt","url"],
            path,
            chunk_size:int =650,
            chunk_overlap:int=70):
        self.doctype = doctype
        self.path = path
        if not all([self.doctype,self.path]):
            raise ValueError("Please pass the required parameters [doctype,embeddings,uploaded_file_path]")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self):
        separators = [
                    "\n\n",              # paragraphs
                    "\nclass ",          # class definitions
                    "\ndef ",            # function definitions
                    "\nif ", "\nfor ", "\nwhile ",  # control blocks
                    "\n\n#",             # comments (Python style)
                    "\n//", "\n/*",      # C/C++/JS comments
                    "\n",                # lines
                    r"\n\d+\.",          # numbered lists
                    r"\n•",              # bullet points
                    ".", "?", "!",       # sentences
                    ";", ":",            # clauses
                    ",",                 # small pauses
                    " ",                 # words
                    ""                   # fallback
                ]
        splitter =  RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=separators,
                is_separator_regex = True
            )
        try:
            if self.doctype in ["pdf","txt"]:
                if self.doctype =="pdf":
                    loader = PyMuPDFLoader(file_path=self.path)
                    doc = loader.load()
                elif self.doctype=="txt":
                    loader = TextLoader(file_path=self.path)
                    doc =  loader.load()
            elif self.doctype=="url":
                path = self.uploaded_file_path
                loader = WebBaseLoader(web_path=self.path)
                doc =   loader.load()

            chunked_docs = splitter.split_documents(doc)
            return chunked_docs

        except Exception as e:
            raise e