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
from langchain.callbacks import get_openai_callback

def main():
    load_dotenv()

    # os.getenv('OPENAI_API_KEY')
    st.write(f"Loaded key: {os.getenv('OPENAI_API_KEY')[:5]}...")
    # UI SETUP
    st.set_page_config(page_title='Ask your PDF')
    st.header('Ask your PDF')
    #upload file
    pdf = st.file_uploader('Upload your pdf', type='pdf')
    
    #extract the text
    if pdf is not None:
        # loader = PyPDFLoader('/Users/felipesilverio/Documents/GitHub/LangChainTest/prova.pdf')
        # docs = loader.load() # 1 doc = 1 page
        # st.write(f'Pages: {len(docs)}')
        reader = PdfReader(pdf)
        pages = reader.pages
        docs = []

        for idx, page in enumerate(pages):
            text = page.extract_text() or ""
            docs.append(Document(page_content=text, metadata={"page": idx+1}))

        st.write(f'Pages: {len(docs)}')
    

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=60, add_start_index=True)
        all_splits = text_splitter.split_documents(docs)

        st.write(all_splits)

        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        vector_store = InMemoryVectorStore(embeddings)
        
        ids = vector_store.add_documents(documents=all_splits)

        user_question = st.text_input("Ask a question about the PDF: ")
        if user_question:   
            results = vector_store.similarity_search_with_score(user_question)
            retrieved_docs = [doc for doc, score in results]

            llm = OpenAI(model="gpt-4o-mini")
            chain = load_qa_chain(llm, chain_type='stuff')
            with get_openai_callback() as cb:
                response = chain.run(input_documents=retrieved_docs, question=user_question)
                st.write(cb)
                st.write(response)

    print('Teste')

if __name__ == '__main__':
    main()