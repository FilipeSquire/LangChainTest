from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from base64 import b64decode
import hashlib
import pickle
import os
from engine2 import EngineJedi
import streamlit as st

def file_md5(path: str) -> str:
    """Compute MD5 hash of a file for caching key."""
    hash_md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def file_hash_from_bytes(data: bytes) -> str:
    """Compute MD5 hash of raw PDF bytes for caching key."""
    hash_md5 = hashlib.md5()
    hash_md5.update(data)
    return hash_md5.hexdigest()


class ShadowEngine():

    def __init__(self, file_pdf):
        load_dotenv()
        self.pdf_file = file_pdf
        self.filename = getattr(file_pdf, 'name', None)
        self.persist_directory = f"./chroma_persist/{self.filename}"

    def _load_pickle(self, name: str):
        path = os.path.join(self.persist_directory, f"{name}.pkl")
        with open(path, 'rb') as f:
            return pickle.load(f)
        
    def remembering(self):

        self.tables = self._load_pickle('tables')          # list of table elements
        self.texts = self._load_pickle('texts')            # list of text elements
        self.table_summaries = self._load_pickle('table_summaries')
        self.text_summaries = self._load_pickle('text_summaries')

    def fit(self):
        # Remembering

        self.remembering()

        imported_engine = EngineJedi(self.pdf_file)

        retriever = imported_engine.vector_store_buildup(self.texts, self.tables, self.text_summaries, self.table_summaries)
        chain, chain_with_sources = imported_engine.RAG_pipeline(retriever)

        return chain, chain_with_sources 
        