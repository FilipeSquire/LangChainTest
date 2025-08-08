import streamlit as st
from langchain.callbacks import get_openai_callback
from engines.hybrid_eng import HybridEngine
from engines.profile_creator_eng import ProfileCreator
from engine1 import EngineApprentice
from engine2 import EngineJedi
import os 
import time

def text_engine(pdf):
    # Fast processing on text only
    st.write('Performing analysis.. ')
    trained_engine = EngineApprentice(pdf)
    trained_engine.fit()
    return trained_engine.vector_store, trained_engine.chain, trained_engine.all_splits

def ocr_engine(pdf):
    # OCR and table parsing
    st.write('Performing OCR ...')
    trained_engine = EngineJedi(pdf)
    chain, chain_with_sources = trained_engine.fit()
    return chain, chain_with_sources 

def company_profile_engine(pdf):
    st.write('Performing OCR ...')
    trained_engine = ProfileCreator(pdf)
    chain, chain_with_sources = trained_engine.fit()
    st.write('Created LangChain ...')
    pdf = trained_engine._create_profile(chain_with_sources)
    st.write('Created Profile')
    return chain, chain_with_sources, pdf

def hybrid_engine(pdf):
    # Show list of available processed PDFs
    # from engine2cached import ShadowEngine
    st.write('Loading processed engine...')
    hydra = HybridEngine(pdf)
    return hydra

OCR_BASE_DIR = "./chroma_persist"
output_placeholder = st.empty()

def main():
    # Title
    st.set_page_config(page_title="TENEO Oraculum")
    st.title("TENEO Oraculum")

    # Description text
    st.write(
        "This project currently has two engines:"
        "\n1st engine is fast and works only on text material, and it is capable of reading pdfs and answering questions upfront."
        "\n2nd engine works on pdfs and is capable of treating text, tables and performing OCR."
        "\n\nLoad your pdf and select the engine. Take in mind that the second engine takes some time to build up intelligence."
        " Otherwise look for the pdfs that already have their interpretation ready to be used."
    )

    # Initialize session state flags
    for key in ('text_mode', 'ocr_mode', 'company_profile_mode', 'processed'):
        if key not in st.session_state:
            st.session_state[key] = False

    # PDF uploader
    pdf_file = st.file_uploader("Upload your PDF", type=["pdf"])

    # Placeholder for messages
    output_placeholder = st.empty()

    # Button controls
    # Create three columns for buttons only
    col1, col2, col3 = st.columns(3)

    # Button controls    
    if col1.button("Text Engine"):
        st.session_state.text_mode = True
        st.session_state.ocr_mode = False
        st.session_state.company_profile_mode = False
        st.session_state.processed = False
        st.session_state.vector_store = None
        st.session_state.chain = None
    
    if col2.button("OCR Engine"):
        st.session_state.ocr_mode = True
        st.session_state.text_mode = False
        st.session_state.company_profile_mode = False
        st.session_state.processed = False
        st.session_state.ocr_chain = None
        st.session_state.ocr_chain_with_sources = None

    if col3.button("Create Company Profile"):
        st.session_state.company_profile_mode = True
        st.session_state.text_mode = False
        st.session_state.ocr_mode = False
        st.session_state.processed = False
        st.session_state.cp_chain = None
        st.session_state.cp_chain_with_sources = None
        st.session_state.init_chat = False
        st.session_state.pdf_file = None


    # TEXT ENGINE FLOW
    if st.session_state.text_mode:
        if not pdf_file:
            st.warning("Please upload a PDF first.")
        else:
            if not st.session_state.processed:
                with st.spinner("Text Engine is processing your PDF. Please wait..."):
                    vector_store, chain, all_splits = text_engine(pdf_file)
                    st.session_state.vector_store = vector_store
                    st.session_state.chain = chain
                    st.write(all_splits)

                output_placeholder.success("Text Engine processing complete!")
                st.session_state.processed = True
            # show question input persistently

            user_question = st.text_input("Ask a question about the PDF:")
            if user_question:
                results = st.session_state.vector_store.similarity_search_with_score(user_question)
                docs = [doc for doc, _ in results]
                # with get_openai_callback() as cb:
                response = st.session_state.chain.run(input_documents=docs, question=user_question)
                # st.write(cb)
                st.write(response)


    # OCR ENGINE FLOW
    if st.session_state.ocr_mode:
        if not pdf_file:
            st.warning("Please upload a PDF first.")
        else:
            if not st.session_state.processed:
                with st.spinner("OCR Engine is analyzing your PDF. This may take a while..."):
                    chain, chain_with_sources = ocr_engine(pdf_file)
                    st.session_state.ocr_chain = chain
                    st.session_state.ocr_chain_with_sources = chain_with_sources

                    output_placeholder.success("OCR Engine processing complete!")
                    
                st.session_state.processed = True


            user_question = st.text_input("Ask a question about the PDF:")
            if user_question:
                # prompt = list_ready_pdfs()
                # with get_openai_callback() as cb:
                results = st.session_state.chain_with_sources.invoke(user_question)
                # st.write(cb)
                st.write(results['response'])


    # COMPANY PROFILE
    if st.session_state.company_profile_mode:
        if not pdf_file:
            st.warning("Please upload a PDF first.")
        else:
            if not st.session_state.processed:
                with st.spinner("Creating company profile .... This may take a while..."):
                    # chain, chain_with_sources, pdf = company_profile_engine(pdf_file)
                    # st.session_state.cp_chain = chain
                    # st.session_state.cp_chain_with_sources = chain_with_sources
                    model2 = hybrid_engine(pdf_file)
                    tic = time.perf_counter()
                    model2.main()
                    st.write(f"OCR Engine processing complete! Total time: {time.perf_counter() - tic}")
                    
                    tic = time.perf_counter()
                    pdf = model2.create_profile()
                    st.write(f"Profile Created! Total time: {time.perf_counter() - tic}")

                    st.download_button(
                        label="Download transformed PDF",
                        data=pdf,
                        file_name="company_profile_demo.pdf",
                        mime="application/pdf",
                    )

                    st.session_state.cp_chain = model2.chain
                    st.session_state.cp_chain_with_sources = model2.chain_with_sources
                
                st.session_state.processed = True

            user_question = st.text_input("Ask a question about the PDF:")
            if user_question:
                # prompt = list_ready_pdfs()
                # with get_openai_callback() as cb:
                results = st.session_state.cp_chain_with_sources.invoke(user_question)
                st.write(results['response'])
                

if __name__ == '__main__':
    main()

