import tempfile

from agent_proto import robust
from fastapi import UploadFile
from langchain.document_loaders.csv_loader import UnstructuredCSVLoader
from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.document_loaders.pdf import UnstructuredPDFLoader
from langchain.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain_core.documents import Document


@robust
async def load_document(file: UploadFile) -> UnstructuredFileLoader:
    """
    Loads a document to an Unstructured json format
    """
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        temp.write(await file.read())
        temp.close()
        assert file.content_type is not None, "Content type is required"
        if "word" in file.content_type:
            return UnstructuredWordDocumentLoader(temp.name)
        if "pdf" in file.content_type:
            return UnstructuredPDFLoader(temp.name)
        if "powerpoint" in file.content_type or "ppt" in file.content_type:
            return UnstructuredPowerPointLoader(temp.name)
        if "excel" in file.content_type:
            return UnstructuredExcelLoader(temp.name)
        if "csv" in file.content_type:
            return UnstructuredCSVLoader(temp.name)
        raise ValueError("Unsupported file type")


@robust
async def process_document(file: UploadFile) -> list[str]:
    """
    Processes a document and returns the result
    """
    return [doc.page_content for doc in (await load_document(file)).load_and_split()]
