
import io
import time
import uuid
import base64
import binascii
from typing import Any, Iterable, List, Optional, Tuple, Dict

from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma  # v0.2 import
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage

from engines.prompts import system_finance_prompt

load_dotenv()


class HybridEngine:
    """Hybrid (dense+sparse) RAG over multi-PDF corpus — optimized.

    Speed-ups:
    - Idempotent build via `_built` guard.
    - Faster partition: keep OCR, disable heavy table structure inference.
    - Summarize only *long* chunks; embed short chunks directly (no LLM call).
    - Lower retrieval fanout (k) and cap images included in prompt.
    - Concurrency for LLM summarization.
    - Stage timings exposed in `self.timings`.
    """

    def __init__(self, pdfs: Optional[Iterable[Tuple[io.BytesIO, str]]] = None) -> None:
        # Inputs
        self._files: List[Tuple[io.BytesIO, str]] = []
        self._built: bool = False
        self.timings: Dict[str, float] = {}

        # Extracted elements
        self.chunks: Optional[List[Any]] = None
        self.tables: List[Any] = []
        self.texts: List[Any] = []
        self.images: List[Any] = []
        self.table_sources: List[str] = []
        self.text_sources: List[str] = []
        self.image_sources: List[str] = []

        # Summaries
        self.text_summaries: List[str] = []
        self.table_summaries: List[str] = []

        # Vector & store
        self.vectorstore = Chroma(collection_name="multi_modal_rag", embedding_function=OpenAIEmbeddings())
        self.store = InMemoryStore()
        self.id_key = "doc_id"
        self.dense_retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key=self.id_key,
            search_kwargs={"k": 12},  # lower fanout
        )

        # Retrievers & chains
        self.hybrid = None
        self.chain = None
        self.chain_with_sources = None

        if pdfs:
            for f_like, name in pdfs:
                self.add_file(f_like, name)

    # ------------------------------ ingestion ------------------------------
    def add_file(self, file_like: io.BytesIO, name: str) -> None:
        try:
            file_like.seek(0)
        except Exception:
            pass
        self._files.append((file_like, name))

    def _unstructured(self) -> None:
        t0 = time.perf_counter()
        all_chunks: List[Any] = []
        tables, texts, images = [], [], []
        t_src, x_src, i_src = [], [], []

        for f_like, fname in self._files:
            f_like.seek(0)
            chunks = partition_pdf(
                file=f_like,
                infer_table_structure=False,   # big speed win; still keeps text
                strategy="hi_res",             # OCR for scanned PDFs
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=6000,
                combine_text_under_n_chars=1500,
                new_after_n_chars=4000,
            )
            all_chunks.extend(chunks)

            for chunk in chunks:
                if hasattr(chunk, "metadata") and getattr(chunk.metadata, "orig_elements", None):
                    for el in chunk.metadata.orig_elements:
                        t = str(type(el))
                        if "Table" in t:
                            tables.append(el); t_src.append(fname)
                        elif "Image" in t:
                            images.append(el); i_src.append(fname)
                        else:
                            texts.append(el); x_src.append(fname)

        self.chunks = all_chunks
        self.tables, self.texts, self.images = tables, texts, images
        self.table_sources, self.text_sources, self.image_sources = t_src, x_src, i_src
        self.timings["unstructured_s"] = time.perf_counter() - t0
        print(f"Finished unstructured — texts={len(texts)} tables={len(tables)} images={len(images)}")

    def _el_text(self, el: Any) -> str:
        return getattr(el, "text", str(el)) or ""

    def _summarization(self) -> None:
        t0 = time.perf_counter()
        prompt_text = (
            "You are a corporate finance expert. Summarize the element below succinctly."
            "Prefer ratios, trends, figures; drop boilerplate."
            "Element: {element}"
        )
        prompt = ChatPromptTemplate.from_template(prompt_text)
        model = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

        # Only summarize long chunks; keep short ones as-is (saves many LLM calls)
        def split_long_short(seq: List[Any], thresh: int = 800):
            long_items, short_items, long_idx, short_idx = [], [], [], []
            for i, el in enumerate(seq):
                s = self._el_text(el)
                if len(s) > thresh:
                    long_items.append(s[:4000])  # truncate for cost
                    long_idx.append(i)
                else:
                    short_items.append(s)
                    short_idx.append(i)
            return long_items, short_items, long_idx, short_idx

        # Texts
        long_t, short_t, long_t_idx, short_t_idx = split_long_short(self.texts)
        summaries_t = [None] * len(self.texts)
        if long_t:
            outs = summarize_chain.batch(long_t, {"max_concurrency": 6})
            for pos, idx in enumerate(long_t_idx):
                summaries_t[idx] = outs[pos]
        for idx, s in zip(short_t_idx, short_t):
            summaries_t[idx] = s
        self.text_summaries = summaries_t  # type: ignore

        # Tables
        long_tb, short_tb, long_tb_idx, short_tb_idx = split_long_short(self.tables, thresh=400)
        summaries_tb = [None] * len(self.tables)
        if long_tb:
            outs = summarize_chain.batch(long_tb, {"max_concurrency": 6})
            for pos, idx in enumerate(long_tb_idx):
                summaries_tb[idx] = outs[pos]
        for idx, s in zip(short_tb_idx, short_tb):
            summaries_tb[idx] = s
        self.table_summaries = summaries_tb  # type: ignore

        self.timings["summarization_s"] = time.perf_counter() - t0
        print("Finished summarization")

    def _store_load(self) -> None:
        t0 = time.perf_counter()
        # Texts → vector + parents
        text_ids = [str(uuid.uuid4()) for _ in self.texts]
        summary_texts = [
            Document(page_content=self.text_summaries[i] or "", metadata={self.id_key: text_ids[i], "source": self.text_sources[i]})
            for i in range(len(self.texts))
        ]
        if summary_texts:
            self.dense_retriever.vectorstore.add_documents(summary_texts)

        parent_text_docs = [
            Document(page_content=self._el_text(el), metadata={"source": self.text_sources[i], "type": "text"})
            for i, el in enumerate(self.texts)
        ]
        if parent_text_docs:
            self.dense_retriever.docstore.mset(list(zip(text_ids, parent_text_docs)))

        # Tables → vector + parents
        table_ids = [str(uuid.uuid4()) for _ in self.tables]
        summary_tables = [
            Document(page_content=self.table_summaries[i] or "", metadata={self.id_key: table_ids[i], "source": self.table_sources[i]})
            for i in range(len(self.tables))
        ]
        if summary_tables:
            self.dense_retriever.vectorstore.add_documents(summary_tables)

        parent_table_docs = [
            Document(page_content=self._el_text(el), metadata={"source": self.table_sources[i], "type": "table"})
            for i, el in enumerate(self.tables)
        ]
        if parent_table_docs:
            self.dense_retriever.docstore.mset(list(zip(table_ids, parent_table_docs)))

        # Images → parents only (cap stored count if huge)
        image_ids = [str(uuid.uuid4()) for _ in self.images]
        parent_image_docs: List[Document] = []
        for i, el in enumerate(self.images):
            b64 = getattr(el, "payload", None) or getattr(el, "data", None) or ""
            if not isinstance(b64, str):
                b64 = str(b64)
            parent_image_docs.append(Document(page_content=b64, metadata={"source": self.image_sources[i], "type": "image"}))
        if parent_image_docs:
            self.dense_retriever.docstore.mset(list(zip(image_ids, parent_image_docs)))

        self.timings["store_load_s"] = time.perf_counter() - t0
        print("Finished store load")

    def _hydra(self) -> None:
        t0 = time.perf_counter()
        # Build BM25 over stored parents
        keys = list(self.store.yield_keys())
        raw_items = self.store.mget(keys)
        parent_docs: List[Document] = []
        for item in raw_items:
            parent_docs.append(item if isinstance(item, Document) else Document(page_content=str(item)))
        bm25 = BM25Retriever.from_documents(parent_docs)
        bm25.k = 12
        self.hybrid = EnsembleRetriever(retrievers=[bm25, self.dense_retriever], weights=[0.4, 0.6])
        self.timings["retriever_build_s"] = time.perf_counter() - t0
        print("Finished hydra")

    # ---------------------------- RAG PIPE ---------------------------------
    def _looks_like_b64(self, s: str) -> bool:
        if not isinstance(s, str) or len(s) < 40:
            return False
        try:
            base64.b64decode(s, validate=True)
            return True
        except binascii.Error:
            return False

    def _to_str(self, obj) -> str:
        if isinstance(obj, str):
            return obj
        if hasattr(obj, "page_content"):
            return obj.page_content
        if hasattr(obj, "text"):
            return obj.text
        return str(obj)

    def _parse_docs(self, docs):
        images, texts = [], []
        for d in docs:
            payload = self._to_str(d)
            (images if self._looks_like_b64(payload) else texts).append(payload)
        return {"images": images[:6], "texts": texts}  # cap images in prompt

    def _build_prompt_two(self, kwargs) -> ChatPromptTemplate:
        ctx = kwargs["context"]; question = kwargs["question"]
        parts = [{"type": "text", "text": f"Context:{ctx.get('texts', [])[:6]}Question: {question}"}]
        for b64 in ctx.get("images", [])[:6]:
            parts.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        messages = [SystemMessage(content=system_finance_prompt), HumanMessage(content=parts)]
        return ChatPromptTemplate.from_messages(messages)

    def _RAG(self) -> None:
        t0 = time.perf_counter()
        gen_model = ChatOpenAI(model="o3")
        self.chain = (
            {"context": self.hybrid | RunnableLambda(self._parse_docs), "question": RunnablePassthrough()}
            | RunnableLambda(self._build_prompt_two)
            | gen_model
            | StrOutputParser()
        )
        self.chain_with_sources = (
            {"context": self.hybrid | RunnableLambda(self._parse_docs), "question": RunnablePassthrough()}
            | RunnablePassthrough().assign(
                response=(RunnableLambda(self._build_prompt_two) | gen_model | StrOutputParser())
            )
        )
        self.timings["rag_build_s"] = time.perf_counter() - t0
        print("Finished RAG")

    # ------------------------- outputs ------------------------------------
    def _generate_pdf(self, text: str) -> bytes:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=letter)
        styles = getSampleStyleSheet()
        body = styles["BodyText"]
        story = []
        for line in (text or "").split(""):
            story.append(Paragraph(line or " ", body))
            story.append(Spacer(1, 4))
        doc.build(story)
        buf.seek(0)
        return buf.read()

    def create_profile(self) -> bytes:
        response = self.chain_with_sources.invoke("Make the company profile. Use the context given")
        return self._generate_pdf(response["response"])  # bytes

    def main(self) -> None:
        if self._built:
            return
        self._unstructured()
        self._summarization()
        self._store_load()
        self._hydra()
        self._RAG()
        self._built = True
        print("Finished pipeline")