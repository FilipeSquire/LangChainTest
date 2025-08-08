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
from jediengine.context_prompt import system_finance_prompt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet

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

    def _build_prompt(self, kwargs):
        """
        Construct a ChatPromptTemplate that always begins with the system prompt,
        then includes context (text + images) and the user question.
        """

        context = kwargs['context']
        question = kwargs['question']

        # Concatenate all text fragments
        context_text = "".join([t.text for t in context.get('texts', [])])

        # Build the messages list: SystemMessage -> HumanMessage
        messages = [
            SystemMessage(content=system_finance_prompt),
            HumanMessage(content=f"Context: {context_text}\nQuestion: {question}")
        ]

        # Include images if present
        for b64 in context.get('images', []):
            messages.append(
                HumanMessage(content={
                    'type': 'image_url',
                    'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
                })
            )

        # Create and return a prompt template from these messages
        return ChatPromptTemplate.from_messages(messages)

    def _RAG_pipeline(self,retriever):

        chain = (
                {
                    'context': retriever | RunnableLambda(self.parse_docs),
                    'question': RunnablePassthrough(),
                }
                | RunnableLambda(self._build_prompt)
                | ChatOpenAI(model='gpt-4o-mini')
                | StrOutputParser()
            )

        chain_with_sources = {
            'context': retriever | RunnableLambda(self.parse_docs),
            'question': RunnablePassthrough(),
            } | RunnablePassthrough().assign(
                response=(
                    RunnableLambda(self._build_prompt)
                    | ChatOpenAI(model='o3')
                    | StrOutputParser()
                )
            )

        return chain, chain_with_sources

    def create_pdf(text, output_path):
        """
        Creates a PDF file from plain text with full Unicode support
        using ReportLab's Platypus framework.
        
        Parameters:
        - text: the string content to write into the PDF.
        - output_path: full file path where the PDF will be saved.
        """
        # 1. Prepare the document
        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        body_style = styles['BodyText']
        
        # 2. Build a "story" of flowable objects
        story = []
        for line in text.split('\n'):
            # Paragraph handles Unicode (e.g. “–”, “é”, emojis, etc.) natively
            story.append(Paragraph(line or ' ', body_style))
            # Small spacer between lines
            story.append(Spacer(1, 4))
        
        # 3. Generate the PDF
        doc.build(story)

    def mod_powerpoint():
        pass

    def fit(self):
        # Remembering

        self.remembering()

        imported_engine = EngineJedi(self.pdf_file)

        retriever = imported_engine.vector_store_buildup(self.texts, self.tables, self.text_summaries, self.table_summaries)
        chain, chain_with_sources = self.RAG_pipeline(retriever)

        my_prompt = 'Make the company profile of Tour Partner Group Limited. Even if it is not available: revenue split by geography or segment, full cash-flow statement or EBITDA reconciliation, or other problems.'
        response = chain_with_sources.invoke(my_prompt)

        return chain, chain_with_sources 
        