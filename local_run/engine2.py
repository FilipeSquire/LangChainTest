from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import uuid
from langchain.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from base64 import b64decode
import hashlib
import pickle
import os

class EngineJedi():

    def __init__(self, file_pdf):
        load_dotenv()
        self.pdf_file = file_pdf
        self.filename = getattr(self.pdf_file, 'name', None)
        self.persist_directory = f"/Users/felipesilverio/Documents/GitHub/LangChainTest/chroma_persist/{self.filename}"

    def _process_text_types(self,chunks):
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
    

    def _run_engine(self):

        #extract the text

        chunks = partition_pdf(
            file= self.pdf_file,
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

        return chunks



    def _summarization(self, texts, tables):

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
        

    def _vector_store_buildup(self, texts, tables, text_summaries, table_summaries):

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

    def parse_docs(self, docs):
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
    
    def build_prompt(self, kwargs):
        
        docs_by_type = kwargs['context']
        user_question = kwargs['question']

        context_text = ""
        if len(docs_by_type['texts']) > 0:
            for text_element in docs_by_type['texts']:
                context_text += text_element.text

        #construct prompt with context
        prompt_template = f""" 
        Answuer the question based only on the following context, which can include text, tables and image.
        Context: {context_text}
        Question: {user_question}
        """

        prompt_content = [{'type':'text','text':prompt_template}]

        if len(docs_by_type['images']) > 0:
            for image in docs_by_type['images']:
                prompt_content.append(
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{image}'},
                    }
                )

        return ChatPromptTemplate.from_messages([HumanMessage(content=prompt_content),])

    def _RAG_pipeline(self,retriever):

        chain = (
                {
                    'context': retriever | RunnableLambda(self.parse_docs),
                    'question': RunnablePassthrough(),
                }
                | RunnableLambda(self.build_prompt)
                | ChatOpenAI(model='gpt-4o-mini')
                | StrOutputParser()
            )

        chain_with_sources = {
            'context': retriever | RunnableLambda(self.parse_docs),
            'question': RunnablePassthrough(),
            } | RunnablePassthrough().assign(
                response=(
                    RunnableLambda(self.build_prompt)
                    | ChatOpenAI(model='o3')
                    | StrOutputParser()
                )
            )

        return chain, chain_with_sources
    
    def _save_preprocessing(self, file, name):
        os.makedirs(self.persist_directory, exist_ok=True)
        path = os.path.join(self.persist_directory, f'{name}.pkl')
        with open(path, 'wb') as f:
            pickle.dump(file, f)

    def fit(self):
        chunks = self._run_engine()
        tables, texts, images = self._process_text_types(chunks)
        text_summaries, table_summaries = self._summarization(texts, tables)        
        self._save_preprocessing(tables,'tables')
        self._save_preprocessing(texts,'texts')
        self._save_preprocessing(text_summaries,'text_summaries')
        self._save_preprocessing(table_summaries,'table_summaries')

        retriever = self._vector_store_buildup(texts, tables, text_summaries, table_summaries)
        chain, chain_with_sources = self._RAG_pipeline(retriever)

        return chain, chain_with_sources 
        