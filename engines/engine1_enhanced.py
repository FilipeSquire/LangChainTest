# engine1.py
from __future__ import annotations

import io
from typing import Iterable, List, Optional, Tuple

from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain.chains.question_answering import load_qa_chain
from langchain_core.language_models.llms import LLM

# Use the official OpenAI SDK Responses API for o-series models (o3)
from openai import OpenAI as OpenAIClient


class OAIResponsesLLM(LLM):
    """LangChain LLM adapter for the Responses API.
    Why: enables `o3` without legacy completions/chat wrappers.
    """

    model: str = "o3"
    reasoning_effort: str = "medium"  # balance cost/latency

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "openai_responses_o_series"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:  # type: ignore[override]
        # `stop` not supported by Responses API; ignored.
        client = OpenAIClient()
        resp = client.responses.create(
            model=self.model,
            input=prompt,
            reasoning={"effort": self.reasoning_effort},
        )
        # Prefer `output_text`; fallback to stitching text blocks.
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text
        parts: List[str] = []
        for item in getattr(resp, "output", []) or []:
            content = getattr(item, "content", [])
            if content and hasattr(content[0], "text") and hasattr(content[0].text, "value"):
                parts.append(content[0].text.value)
        return "\n".join(parts) if parts else str(resp)


class EngineApprentice:
    """Minimal text-only engine with multi-PDF support using `o3`."""

    def __init__(self, files: Optional[Iterable[Tuple[io.BytesIO, str]]] = None) -> None:
        load_dotenv()
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.llm = OAIResponsesLLM(model="o3", reasoning_effort="medium")  # o3 set directly
        self.chain = load_qa_chain(self.llm, chain_type="stuff")

        self._files: List[Tuple[io.BytesIO, str]] = []
        self._ingested_docs: List[Document] = []
        self.documents_ids: List[str] = []
        self.all_splits: List[Document] = []

        if files:
            for f_like, name in files:
                self.add_file(f_like, name)

    def add_file(self, file_like: io.BytesIO, name: str) -> None:
        # Why: callers may reuse buffer; ensure consistent start.
        try:
            file_like.seek(0)
        except Exception:
            pass
        self._files.append((file_like, name))

    def _extract_documents(self) -> List[Document]:
        docs: List[Document] = []
        total_pages = 0
        for f_like, name in self._files:
            reader = PdfReader(f_like)
            pages = reader.pages
            total_pages += len(pages)
            for idx, page in enumerate(pages):
                text = page.extract_text() or ""
                if not text.strip():
                    continue
                docs.append(
                    Document(page_content=text, metadata={"source": name, "page": idx + 1})
                )
        st.write(f"Total pages (non-empty counted per page text): {len(docs)} from {total_pages} raw pages")
        return docs

    def run_engine(self) -> None:
        if not self._files:
            return
        docs = self._extract_documents()
        if not docs:
            return
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400, add_start_index=True)
        self.all_splits = splitter.split_documents(docs)
        self.documents_ids = self.vector_store.add_documents(documents=self.all_splits)

    def fit(self) -> None:
        self.run_engine()

    def query(self, question: str, k: int = 4) -> str:
        docs = [doc for doc, _ in self.vector_store.similarity_search_with_score(question, k=k)]
        if not docs:
            return "No matching context found. Try rephrasing your question."
        return self.chain.run(input_documents=docs, question=question)
