import streamlit as st
from langchain.callbacks import get_openai_callback
import os 
from engine2 import EngineJedi
from engine1 import EngineApprentice
from engine2cached import ShadowEngine

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


def list_ready_pdfs(pdf):
    # Show list of available processed PDFs
    st.write('Loading processed engine...')
    shadow_engine = ShadowEngine(pdf)
    chain, chain_with_sources = shadow_engine.fit()
    return chain, chain_with_sources 

OCR_BASE_DIR = "/Users/felipesilverio/Documents/GitHub/LangChainTest/chroma_persist"
output_placeholder = st.empty()

def main():
    # Title
    st.set_page_config(page_title="LGG Brain")
    st.title("LGG Brain")

    # Description text
    st.write(
        "This project currently has two engines:"
        "\n1st engine is fast and works only on text material, and it is capable of reading pdfs and answering questions upfront."
        "\n2nd engine works on pdfs and is capable of treating text, tables and performing OCR."
        "\n\nLoad your pdf and select the engine. Take in mind that the second engine takes some time to build up intelligence."
        " Otherwise look for the pdfs that already have their interpretation ready to be used."
    )

    # Initialize session state flags
    for key in ('text_mode', 'ocr_mode', 'ready_mode', 'processed'):
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
    col1, col2, col3 = st.columns(3)
    if col1.button("Text Engine"):
        st.session_state.text_mode = True
        st.session_state.ocr_mode = False
        st.session_state.ready_mode = False
        st.session_state.processed = False
        st.session_state.vector_store = None
        st.session_state.chain = None
    
    if col2.button("OCR Engine"):
        st.session_state.ocr_mode = True
        st.session_state.text_mode = False
        st.session_state.ready_mode = False
        st.session_state.processed = False
        st.session_state.new_chain = None
        st.session_state.new_chain_with_sources = None

    if col3.button("Ready-to-go PDFs"):
        st.session_state.ready_mode = True
        st.session_state.text_mode = False
        st.session_state.ocr_mode = False
        st.session_state.processed = False
        st.session_state.chain = None
        st.session_state.chain_with_sources = None
        st.session_state.init_chat = False
        st.session_state.download_chat = False
        st.session_state.pdf_file = None


    # TEXT ENGINE FLOW
    if st.session_state.text_mode:
        if not pdf_file:
            st.warning("Please upload a PDF first.")
        else:
            if not st.session_state.processed:
                with st.spinner("Text Engine is processing your PDF. Please wait..."):
                    st.write('Engine initialized \n Testing version')

                output_placeholder.success("Text Engine processing complete!")
                st.session_state.processed = True
            # show question input persistently

            user_question = st.text_input("Ask a question about the PDF:")
            if user_question:
                st.write('Question registered \n Question computed by model \n Output to be received \n Testing version')




    # OCR ENGINE FLOW
    if st.session_state.ocr_mode:
        if not pdf_file:
            st.warning("Please upload a PDF first.")
        else:
            if not st.session_state.processed:
                with st.spinner("OCR Engine is analyzing your PDF. This may take a while..."):
                    st.write('The engine is starting and performing the processing. It takes some time. \n Testing mode.')
                    chain, chain_with_sources  = ocr_engine(pdf_file)
                    st.session_state.new_chain = chain
                    st.session_state.new_chain_with_sources = chain_with_sources

                output_placeholder.success("OCR Engine processing complete!")
                st.session_state.processed = True

            user_question = st.text_input("Ask a question about the PDF:")
            if user_question:
                st.write('The question has been captured and sent to model. \n Output will be listed here \n Testing mode.')
                # prompt = list_ready_pdfs()
                with get_openai_callback() as cb:
                    results = st.session_state.new_chain_with_sources.invoke(user_question)
                    st.write(cb)
                    st.write(results['response'])
            # downstream OCR UI here

    # READY-TO-GO FLOW
    if st.session_state.ready_mode:

        try:
            folders = [d for d in os.listdir(OCR_BASE_DIR) if os.path.isdir(os.path.join(OCR_BASE_DIR, d))]
        except FileNotFoundError:
            folders = []
            st.error(f"Directory '{OCR_BASE_DIR}' not found.")

        st.subheader("Select a processed PDF folder:")
        for folder in folders:
            if st.button(folder):
                selected = os.path.join(OCR_BASE_DIR, folder)
                st.session_state['pdf_file'] = selected
                st.session_state['processed'] = False
                output_placeholder.success(f"PDF in memory: {folder}")

            if st.session_state.pdf_file:
                if st.button('Initialize chat'):
                    st.session_state['init_chat'] = True
                    st.session_state['download_chat'] = False
                if st.session_state['init_chat']:
                    st.write('Session being built...')
                    user_question = st.text_input("Ask a question about the PDF:")
                    if user_question:
                        # prompt = list_ready_pdfs()
                        st.write('Question registered')
                    
                if st.button('Download Company Profile'):
                    st.session_state['init_chat'] = False
                    st.session_state['download_chat'] = True
                if st.session_state['download_chat']:
                    st.write('Session being built...')
                    
            if not st.session_state.processed:
                with st.spinner("OCR Engine is analyzing your PDF. This may take a while..."):
                    st.write('Imported vectorstore')
                output_placeholder.success("OCR Engine processing complete!")
                st.session_state.processed = True

        # display list UI here

if __name__ == '__main__':
    main()

