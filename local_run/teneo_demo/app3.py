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
from unstructured.partition.pdf import partition_pdf
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

from base64 import b64decode

def process_pdf(pdf):

    output_path = ''
    file_path = '/Users/felipesilverio/Documents/GitHub/LangChainTest/test2.pdf'

    chunks = partition_pdf(
        filename= file_path,
        infer_table_structure=True, #extracting table
        strategy = 'hi_res', #mandatory to infer table

        extract_image_block_types=['Image'], #add 'Table' to list to extract image of tables
        # image_output_dir_path = output_path, #if None, images and tables will be saved as base64

        extract_image_block_to_payload=True, #if true, extract base64 for API usage

        chunking_strategy='by_title', #or basic
        max_characters=10000, #default is 500
        combine_text_under_n_chars=2000, #default is 0
        new_after_n_chars=6000, #default is 0
    )
    
    tables, texts, images= [], [], []

    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Table" in str(type(el)):
                    tables.append(el)
                elif "Image" in str(type(el)):
                    images.append(el)
                else:
                    texts.append(el)

    return tables, texts, images

def summarization(texts, tables):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt_text = """
    You are an especialist in corporate finance tasked with summarizing the text, tables and images.
    Give a concise summary of the table, text or image.
    The tables will be received in format html. Transform this format in order to interpret the table.

    The summary must take special attention to financial-related numbers and statistics, such as monthly or yearly comparisons, debt/loan information, and other subjects related.
    The summary must contain the numerical information of debt, loan, revenue, deficit, and other related topics.

    Response only with the summary, no additional comment. 
    Do not start your message by saying "Here is a summary" or anything like that. 
    Just give the summart as it is.

    Table or text chunk: {element}
    """

    prompt = ChatPromptTemplate.from_template(prompt_text)

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.5,)
    # model = ChatGroq(temperature=0.5, model='llama-3.1-8b-instant')
    summarize_chain = {"element": lambda x: x} | prompt | model | StrOutputParser()

    text_summaries = summarize_chain.batch(texts, {'max_concurrency':3})

    # tables_html = [table.metadata.text_as_html for table in tables]

    table_summaries = summarize_chain.batch(tables, {'max_concurrency':3})

    return text_summaries, table_summaries

def vector_store_buildup(texts, tables, text_summaries, table_summaries):
    import uuid
    from langchain.vectorstores import Chroma
    from langchain.storage import InMemoryStore
    from langchain_core.documents import Document
    from langchain_openai import OpenAIEmbeddings
    from langchain.retrievers.multi_vector import MultiVectorRetriever

    vectorstore = Chroma(collection_name='multi_modal_rag', embedding_function=OpenAIEmbeddings())

    #Storage layor
    store = InMemoryStore()
    id_key = 'doc_id'

    #retriever

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key,
    )

    #Loading values

    doc_ids = [str(uuid.uuid4()) for _ in texts]
    summary_texts = [
        Document(page_content=summary, metadata={id_key: doc_ids[i]}) for i, summary in enumerate(text_summaries)
    ]
    retriever.vectorstore.add_documents(summary_texts)
    retriever.docstore.mset(list(zip(doc_ids, texts)))

    # Add tables
    tables_ids = [str(uuid.uuid4()) for _ in tables]
    summary_tables = [
        Document(page_content=summary, metadata={id_key: tables_ids[i]}) for i, summary in enumerate(table_summaries)
    ]
    retriever.vectorstore.add_documents(summary_tables)
    retriever.docstore.mset(list(zip(tables_ids, tables)))
    
    return retriever


def parse_docs(docs):
    from base64 import b64decode
    # Split base64 images and texts
    b64 = []
    text = []
    for doc in docs:
        try:
            b64decode(doc)
            b64.append(doc)
        except Exception as e:
            text.append(doc)
    return{'images':b64, 'texts':text}

def RAG_pipeline(retriever, build_prompt)

    chain = (
        {
            'context': retriever | RunnableLambda(parse_docs),
            'question': RunnablePassthrough(),
        }
        | RunnableLambda(build_prompt)
        | ChatOpenAI(model='gpt-4o-mini')
        | StrOutputParser()
    )

    chain_with_sources = {
        'context': retriever | RunnableLambda(parse_docs),
        'question': RunnablePassthrough(),
    } | RunnablePassthrough().assign(
        response=(
            RunnableLambda(build_prompt)
            | ChatOpenAI(model='gpt-4o-mini')
            | StrOutputParser()
        )
    )

    return chain, chain_with_sources

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