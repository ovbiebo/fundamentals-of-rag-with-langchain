import streamlit as st
from langchain import hub
from langchain.text_splitter import CharacterTextSplitter
from langchain_qdrant import Qdrant
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
import pymupdf


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def read_pdf(file):
    pdf_document = pymupdf.open(stream=file.read(), filetype="pdf")
    text = ""
    for page_num in range(pdf_document.page_count):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text


def generate_response(uploaded_files, openai_api_key, query_text):
    documents = []
    for uploaded_file in uploaded_files:
        # Check if the file is a PDF
        if uploaded_file.type == "application/pdf":
            document_text = read_pdf(uploaded_file)
        else:
            document_text = uploaded_file.read().decode()
        documents.append(document_text)

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    texts = []
    for document in documents:
        texts.extend(text_splitter.create_documents([document]))

    llm = ChatOpenAI(model="gpt-4o-mini", api_key=openai_api_key)
    # Select embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_api_key)
    # Create a vectorstore from documents
    qdrant = Qdrant.from_documents(
        texts,
        embeddings,
        location=":memory:",  # Local mode with in-memory storage only
        collection_name="my_documents",
    )
    # Create retriever interface
    retriever = qdrant.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")
    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    # Create QA chain
    response = rag_chain.invoke(query_text)
    return response


# File upload
uploaded_file = st.file_uploader('Upload one or more articles', type=['txt', 'pdf'], accept_multiple_files=True)
# Query text
query_text = st.text_input('Enter your question:', placeholder='Please provide a short summary.',
                           disabled=not uploaded_file)

# Form input and query
result = None
with st.form('myform', clear_on_submit=False, border=False):
    openai_api_key = st.text_input('OpenAI API Key', type='password', disabled=not (uploaded_file and query_text))
    submitted = st.form_submit_button('Submit', disabled=not (uploaded_file and query_text))
    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result = response
if result:
    st.info(result)
