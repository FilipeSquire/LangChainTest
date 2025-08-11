from io import BytesIO
import time
from typing import List, Tuple

import streamlit as st
from langchain.callbacks import get_openai_callback

from engines.hybrid_eng import HybridEngine
from engines.profile_creator_eng import ProfileCreator
from engines.engine1_enhanced import EngineApprentice
from engines.engine2 import EngineJedi


# ---- Cached builders -------------------------------------------------------
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


@st.cache_resource(show_spinner=False)
def ocr_engine_cached_single(pdf_bytes: bytes):
    pdf_stream = BytesIO(pdf_bytes)
    engine = EngineJedi(pdf_stream)
    chain, chain_with_sources = engine.fit()
    return chain, chain_with_sources


@st.cache_resource(show_spinner=False)
def profile_engine_cached_single(pdf_bytes: bytes):
    pdf_stream = BytesIO(pdf_bytes)
    engine = HybridEngine(pdf_stream)

    tic = time.perf_counter()
    engine.main()
    st.write(f"OCR Engine processing complete! Total time: {time.perf_counter() - tic}")

    tic = time.perf_counter()
    pdf_out = engine.create_profile()
    st.write(f"Profile Creation Completed! Total time: {time.perf_counter() - tic}")
    return engine.chain, engine.chain_with_sources, pdf_out


# ---- App -------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Teneo Oraculum")
    st.title("GT Oraculum")

    st.write(
        "Available engines:\n"
        "1. Text Engine (fast, text-only; now supports multiple PDFs)\n"
        "2. OCR Engine (OCR + tables; demo uses one file at a time)\n"
        "3. Company Profile Creator (full hybrid; one file at a time)\n"
        "\nUpload one or more PDFs and select an engine."
    )

    # Initialize session flags
    for key in ("text_mode", "ocr_mode", "company_profile_mode", "processed"):
        if key not in st.session_state:
            st.session_state[key] = False

    pdf_files = st.file_uploader("Upload your PDF(s)", type=["pdf"], accept_multiple_files=True)
    output_placeholder = st.empty()
    col1, col2, col3 = st.columns(3)

    if col1.button("Text Engine"):
        st.session_state.update({
            "text_mode": True,
            "ocr_mode": False,
            "company_profile_mode": False,
            "processed": False,
        })

    if col2.button("OCR Engine"):
        st.session_state.update({
            "ocr_mode": True,
            "text_mode": False,
            "company_profile_mode": False,
            "processed": False,
        })

    if col3.button("Create Company Profile"):
        st.session_state.update({
            "company_profile_mode": True,
            "text_mode": False,
            "ocr_mode": False,
            "processed": False,
        })

    # --- Text Engine (multi-file) ------------------------------------------
    if st.session_state.text_mode and pdf_files:
        files_bytes: Tuple[bytes, ...] = tuple(f.read() for f in pdf_files)
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

    # --- OCR Engine (single-file demo, can be looped) ----------------------
    if st.session_state.ocr_mode and pdf_files:
        if len(pdf_files) != 1:
            st.info("For now, OCR mode processes one file at a time. Select a single file.")
        else:
            pdf_bytes = pdf_files[0].read()
            if not st.session_state.processed:
                with st.spinner("Running OCR engine..."):
                    chain, chain_src = ocr_engine_cached_single(pdf_bytes)
                    st.session_state.ocr_chain = chain
                    st.session_state.ocr_chain_with_sources = chain_src
                output_placeholder.success("OCR Engine done.")
                st.session_state.processed = True

            question = st.text_input("Ask a question:")
            if question:
                res = st.session_state.ocr_chain_with_sources.invoke(question)
                st.write(res["response"])

    # --- Company Profile (single-file demo, can be looped) -----------------
    if st.session_state.company_profile_mode and pdf_files:
        if len(pdf_files) != 1:
            st.info("For now, Company Profile mode processes one file at a time. Select a single file.")
        else:
            pdf_bytes = pdf_files[0].read()
            if not st.session_state.processed:
                with st.spinner("Creating company profile..."):
                    cp_chain, cp_chain_src, out_pdf = profile_engine_cached_single(pdf_bytes)
                    st.session_state.cp_chain = cp_chain
                    st.session_state.cp_chain_with_sources = cp_chain_src
                    st.download_button(
                        "Download Profile PDF",
                        data=out_pdf,
                        file_name="company_profile.pdf",
                        mime="application/pdf",
                    )
                output_placeholder.success("Profile creation done.")
                st.session_state.processed = True

            question = st.text_input("Ask a question:")
            if question:
                res = st.session_state.cp_chain_with_sources.invoke(question)
                st.write(res["response"]) 


if __name__ == "__main__":
    main()