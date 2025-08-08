from io import BytesIO
import streamlit as st
from langchain.callbacks import get_openai_callback
from engines.hybrid_eng import HybridEngine
from engines.profile_creator_eng import ProfileCreator
from engine1 import EngineApprentice
from engine2 import EngineJedi
import os
import time 
# Use cache_resource for unpickleable engine objects
@st.cache_resource
def text_engine_cached(pdf_bytes: bytes):
    """Process PDF text with EngineApprentice (cached resource)."""
    pdf_stream = BytesIO(pdf_bytes)
    engine = EngineApprentice(pdf_stream)
    engine.fit()
    return engine.vector_store, engine.chain, engine.all_splits

@st.cache_resource
def ocr_engine_cached(pdf_bytes: bytes):
    """Perform OCR and table parsing with EngineJedi (cached resource)."""
    pdf_stream = BytesIO(pdf_bytes)
    engine = EngineJedi(pdf_stream)
    chain, chain_with_sources = engine.fit()
    return chain, chain_with_sources


@st.cache_resource
def profile_engine_cached(pdf_bytes: bytes):
    """Create company profile via HybridEngine (cached)."""
    pdf_stream = BytesIO(pdf_bytes)
    engine = HybridEngine(pdf_stream)
    
    tic = time.perf_counter()
    engine.main()
    st.write(f"OCR Engine processing complete! Total time: {time.perf_counter() - tic}")

    tic = time.perf_counter()
    pdf_out = engine.create_profile()
    st.write(f"Profile Creation Completed! Total time: {time.perf_counter() - tic}")
    return engine.chain, engine.chain_with_sources, pdf_out

# Streamlit App
def main():
    st.set_page_config(page_title="Teneo Oraculum")
    st.title("Teneo Oraculum")

    st.write(
        "Available engines:\n"
        "1. Text Engine (fast, text-only)\n"
        "2. OCR Engine (OCR + tables)\n"
        "3. Company Profile Creator (full hybrid)\n"
        "\nUpload your PDF and select an engine. Expensive engines are cached to speed up repeat runs."
    )

    # Initialize session flags
    for key in ('text_mode', 'ocr_mode', 'company_profile_mode', 'processed'):
        if key not in st.session_state:
            st.session_state[key] = False

    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])
    output_placeholder = st.empty()
    col1, col2, col3 = st.columns(3)

    if col1.button("Text Engine"):
        st.session_state.update({
            'text_mode': True,
            'ocr_mode': False,
            'company_profile_mode': False,
            'processed': False
        })

    if col2.button("OCR Engine"):
        st.session_state.update({
            'ocr_mode': True,
            'text_mode': False,
            'company_profile_mode': False,
            'processed': False
        })

    if col3.button("Create Company Profile"):
        st.session_state.update({
            'company_profile_mode': True,
            'text_mode': False,
            'ocr_mode': False,
            'processed': False
        })

    if pdf_file:
        pdf_bytes = pdf_file.read()

        # Text Engine
        if st.session_state.text_mode:
            if not st.session_state.processed:
                with st.spinner("Processing text engine..."):
                    vs, chain, splits = text_engine_cached(pdf_bytes)
                    st.session_state.vector_store = vs
                    st.session_state.chain = chain
                    st.write(splits)
                output_placeholder.success("Text Engine done.")
                st.session_state.processed = True

            question = st.text_input("Ask a question:")
            if question:
                docs = [doc for doc, _ in st.session_state.vector_store.similarity_search_with_score(question)]
                response = st.session_state.chain.run(input_documents=docs, question=question)
                st.write(response)

        # OCR Engine
        if st.session_state.ocr_mode:
            if not st.session_state.processed:
                with st.spinner("Running OCR engine..."):
                    chain, chain_src = ocr_engine_cached(pdf_bytes)
                    st.session_state.ocr_chain = chain
                    st.session_state.ocr_chain_with_sources = chain_src
                output_placeholder.success("OCR Engine done.")
                st.session_state.processed = True

            question = st.text_input("Ask a question:")
            if question:
                res = st.session_state.ocr_chain_with_sources.invoke(question)
                st.write(res['response'])

        # Company Profile Engine
        if st.session_state.company_profile_mode:
            if not st.session_state.processed:
                with st.spinner("Creating company profile..."):
                    cp_chain, cp_chain_src, out_pdf = profile_engine_cached(pdf_bytes)
                    st.session_state.cp_chain = cp_chain
                    st.session_state.cp_chain_with_sources = cp_chain_src
                    st.download_button(
                        "Download Profile PDF", data=out_pdf,
                        file_name="company_profile.pdf", mime="application/pdf"
                    )
                output_placeholder.success("Profile creation done.")
                st.session_state.processed = True

            question = st.text_input("Ask a question:")
            if question:
                res = st.session_state.cp_chain_with_sources.invoke(question)
                st.write(res['response'])

if __name__ == '__main__':
    main()
