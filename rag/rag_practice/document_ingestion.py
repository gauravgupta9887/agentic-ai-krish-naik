from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path


def get_pdf_document():
    pdf_path = Path("Information_Retrieval_and_Text_Processing_Guide_6308.pdf")
    loader = PyMuPDFLoader(str(pdf_path))
    docs = loader.load()

    return docs
