from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
import os

# from langchain.text_splitter import CharacterTextSplitter
# from langchain_openai import OpenAIEmbeddings
# from langchain_community.vectorstores import FAISS
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
# from langchain.llms import openai

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.llms import OpenAI  # or ChatOpenAI

from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

import tiktoken

def estimate_embedding_cost(text, model="text-embedding-ada-002", price_per_1k=0.00002):
    tokenizer = tiktoken.encoding_for_model(model)
    tokens = len(tokenizer.encode(text, disallowed_special=()))
    cost = tokens * price_per_1k / 1000
    return tokens, round(cost, 8)

def main():
    load_dotenv()
    # os.getenv('OPENAI_API_KEY')
    st.write(f"Loaded key: {os.getenv('OPENAI_API_KEY')[:5]}...")
    # UI SETUP
    MAX_TOKENS = 8_000      # ≈ cost ceiling you accept per PDF
    MAX_CHUNKS = 200        # guard‑rail against giant docs

    st.set_page_config(page_title='Ask your PDF')
    st.header('Ask your PDF')

    #upload file
    pdf = st.file_uploader('Upload your pdf', type='pdf')

    #extract the text
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        # split into chunks 
        text_splitter = CharacterTextSplitter(
            separator=' ',
            chunk_size=400, #amount of characters
            chunk_overlap = 60, #if the chunk_size ends in a middle of a sentence. The next chink starts 200 characters before the end of chunk_size
            length_function = len
        )

        chunks = text_splitter.split_text(text)[:MAX_CHUNKS]

        tokens, cost = estimate_embedding_cost(text)
        st.write(f"🔢 Tokens: {tokens}")
        st.write(f"💵 Estimated embedding cost: ${cost}")

        #create embeddings
        # FAISS is a lib developed by Facebook that allows to do similarity search, its an AI model
        try:
            embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            # vectors = embeddings.embed_documents(chunks)

            knowledge_base = FAISS.from_texts(chunks, embeddings)
        except Exception as e:
            st.write(e)

        # UI SETUP
        user_question = st.text_input("Ask a question about the PDF: ")

        if user_question:
            try:
                docs = knowledge_base.similarity_search(user_question)

                llm = OpenAI()
                chain = load_qa_chain(llm, chain_type='stuff')
                with get_openai_callback() as cb:
                    response = chain.run(input_documents=docs, question=user_question)
                    st.write(cb)
                st.write(response)
            except Exception as e:
                st.write('An error happened in the embeddings section')

        
        st.write(chunks)

    print('Tetse')

if __name__ == '__main__':
    main()