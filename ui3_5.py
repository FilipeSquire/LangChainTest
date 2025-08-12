from io import BytesIO
from typing import Tuple
import time
import streamlit as st

from engines.hybrig_eng_enhanced import HybridEngine
from engines.engine1_enhanced import EngineApprentice

# ---- Cached builders -------------------------------------------------------
@st.cache_resource(show_spinner=False)
def ocr_engine_cached_multi(files_bytes: Tuple[bytes, ...], files_names: Tuple[str, ...]):
    """Multi-file OCR mode via HybridEngine (idempotent + timed)."""
    pdf_streams = tuple((BytesIO(b), n) for b, n in zip(files_bytes, files_names))
    engine = HybridEngine(pdf_streams)
    t0 = time.perf_counter(); engine.main(); build_s = time.perf_counter() - t0
    timings = getattr(engine, "timings", {})
    timings["total_build_s"] = build_s
    return engine.chain, engine.chain_with_sources, timings


@st.cache_resource(show_spinner=False)
def text_engine_cached_multi(files_bytes: Tuple[bytes, ...], files_names: Tuple[str, ...]):
    """Build a single EngineApprentice over many PDFs (cached).

    Why tuples: make args hashable for Streamlit's cache.
    """
    engine = EngineApprentice()
    for b, name in zip(files_bytes, files_names):
        engine.add_file(BytesIO(b), name)
    engine.fit()
    return engine.vector_store, engine.chain, engine.all_splits


# ---- App -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Oraculum")
    st.title("Oraculum")
    output_placeholder = st.empty()

    st.write(
        "Available engines: \n"
        "1. Text Engine (fast, text-only; multi-PDF) \n"
        "2. OCR Engine (hybrid OCR; multi-PDF)\n"
        "Upload one or more PDFs and select an engine."
    )

    for key in ("ocr_mode", "text_engine", "processed"):
        if key not in st.session_state:
            st.session_state[key] = False

    pdf_files = st.file_uploader("Upload your PDF(s)", type=["pdf"], accept_multiple_files=True)
    col1, col2 = st.columns(2)

    if col1.button("OCR Engine"):
        st.session_state.update({"ocr_mode": True, "text_engine": False, "processed": False})
    if col2.button("Text Engine"):
        st.session_state.update({"text_engine": True, "ocr_mode": False, "processed": False})

    # --- OCR Engine (multi-file) -------------------------------------------
    if st.session_state.ocr_mode and pdf_files:
        # IMPORTANT: use getvalue() so cache keys are stable across reruns
        files_bytes: Tuple[bytes, ...] = tuple(f.getvalue() for f in pdf_files)
        files_names: Tuple[str, ...] = tuple(f.name for f in pdf_files)

        if not st.session_state.processed:
            with st.spinner("Building OCR (hybrid) index across all PDFs..."):
                chain, chain_src, timings = ocr_engine_cached_multi(files_bytes, files_names)
                st.session_state.ocr_chain = chain
                st.session_state.ocr_chain_with_sources = chain_src
                st.session_state.ocr_timings = timings
            st.success("OCR index ready.")
            st.session_state.processed = True

        st.subheader("Timings")
        st.json(st.session_state.ocr_timings)

        question = st.text_input("Ask a question about your PDFs:")
        if question:
            res = st.session_state.ocr_chain_with_sources.invoke(question)
            st.write(res["response"])

    # --- Company Profile (multi-file) --------------------------------------
    if st.session_state.text_engine and pdf_files:
        files_bytes: Tuple[bytes, ...] = tuple(f.getvalue() for f in pdf_files)
        files_names: Tuple[str, ...] = tuple(f.name for f in pdf_files)

        if not st.session_state.processed:
            with st.spinner("Processing text engine across all PDFs..."):
                vs, chain, splits = text_engine_cached_multi(files_bytes, files_names)
                st.session_state.vector_store = vs
                st.session_state.chain = chain
                st.session_state.splits = splits
                st.write({"files": files_names, "chunks": len(splits)})
            output_placeholder.success("Text Engine done.")
            st.session_state.processed = True

        question = st.text_input("Ask a question across your library:")
        if question:
            docs = [doc for doc, _ in st.session_state.vector_store.similarity_search_with_score(question)]
            response = st.session_state.chain.run(input_documents=docs, question=question)
            st.write(response)
            if docs:
                with st.expander("Context (top matches)"):
                    for d in docs:
                        meta = d.metadata or {}
                        st.caption(f"{meta.get('source', 'unknown')} — page {meta.get('page', '?')}")
                        st.text(d.page_content[:800])


if __name__ == "__main__":
    main()