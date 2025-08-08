from dotenv import load_dotenv
import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.llms import OpenAI  # or ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document
from PyPDF2 import PdfReader

class EngineApprentice():

    def __init__(self, file_pdf):
        load_dotenv()

        self.pdf = file_pdf
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.documents_ids = []
        self.llm = OpenAI(model="gpt-4o-mini")
        self.chain = load_qa_chain(self.llm, chain_type='stuff')
        self.all_splits = []

    def run_engine(self):

        #extract the text
        if self.pdf is not None:
            # loader = PyPDFLoader('/Users/felipesilverio/Documents/GitHub/LangChainTest/prova.pdf')
            # docs = loader.load() # 1 doc = 1 page
            # st.write(f'Pages: {len(docs)}')
            reader = PdfReader(self.pdf)
            pages = reader.pages
            docs = []

            for idx, page in enumerate(pages):
                text = page.extract_text() or ""
                docs.append(Document(page_content=text, metadata={"page": idx+1}))

            st.write(f'Size of the file: {len(docs)} pages')


            text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60, add_start_index=True)
            all_splits = text_splitter.split_documents(docs)
            self.all_splits = all_splits
            # st.write(all_splits)
            ids = self.vector_store.add_documents(documents=all_splits)
            self.documents_ids = ids


    def fit(self):
        self.run_engine()