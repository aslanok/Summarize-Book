import streamlit as st
import markdown2
import pdfplumber
import docx
from io import StringIO
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from chromadb import Client, Settings
from uuid import uuid4
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
import os

def read_pdf(file):
    pdf = pdfplumber.open(file)
    text = ""
    for page in pdf.pages:
        text += page.extract_text()
    pdf.close()
    return text

def read_docx(file):
    doc = docx.Document(file)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def read_txt(file):
    stringio = StringIO(file.getvalue().decode("utf-8"))
    text = stringio.read()
    return text

def process_file(file):
    if file.name.endswith(".pdf"):
        return read_pdf(file)
    elif file.name.endswith(".docx"):
        return read_docx(file)
    elif file.name.endswith(".txt"):
        return read_txt(file)
    elif file.name.endswith(".md"):
        return markdown2.markdown(file.getvalue().decode("utf-8"))
    else:
        st.error("Unsupported file type!")
        return None

# Initialize Streamlit app
st.title("Text Summarization and QA with ChromaDB")

# Upload file (multiple formats supported)
uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt", "md"])

if uploaded_file is not None:
    # Process the file based on its type
    text_content = process_file(uploaded_file)
    if text_content is None:
        st.stop()  # Stop execution if the file type is unsupported

    # Wrap the text in a Document object
    documents = [Document(page_content=text_content, metadata={"source": uploaded_file.name})]

    # Generate a unique identifier for the document
    document_id = str(uuid4())

    # Function to split text
    def split_text(documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )
        chunks = text_splitter.split_documents(documents)
        st.write(f"Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    # Split the text into chunks
    chunks = split_text(documents)

    # Load embedding model
    embeddingModel = SentenceTransformer('all-MiniLM-L6-v2')

    # Embed the text chunks
    chunk_embeddings = []
    for chunk in chunks:
        embedding = embeddingModel.encode(chunk.page_content)
        chunk_embeddings.append({
            "embedding": embedding,
            "metadata": chunk.metadata,
            "content": chunk.page_content
        })

    # Initialize ChromaDB client and collection using Streamlit's session state
    if "chroma_client" not in st.session_state:
        # Create a new client instance and store it in session state
        chroma_dir = f'./chromadb_data/{document_id}'
        os.makedirs(chroma_dir, exist_ok=True)
        st.session_state.chroma_client = Client(Settings(persist_directory=chroma_dir))
        st.session_state.chroma_collection_name = f'doc_{document_id}'
        st.session_state.chroma_collection = st.session_state.chroma_client.create_collection(st.session_state.chroma_collection_name)
    else:
        chroma_dir = f'./chromadb_data/{document_id}'

    # Add embeddings and metadata to the collection stored in session state
    for chunk_data in chunk_embeddings:
        doc_id = str(uuid4())
        embedding_list = chunk_data["embedding"].tolist()
        st.session_state.chroma_collection.add(
            ids=[doc_id],
            documents=[chunk_data["content"]],
            embeddings=[embedding_list],
            metadatas=[chunk_data["metadata"]]
        )

    # Initialize the LLM and QA chain
    llm = Ollama(model="gemma2:2b")
    prompt = """
    1. Use the following context to answer the question at the end.
    2. Be precise and avoid speculation. If the information isn't clear, say "I don't know."
    3. Provide a concise, 2-3 sentence answer.

    Context: {context}

    Question: {question}

    Accurate Answer:"""

    QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt)

    # Manually create the sequence
    def manual_llm_chain(llm, prompt_template, question):
        context = prompt_template.format(context=question["context"], question=question["question"])
        return llm(context)

    embedding_model = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=chroma_dir,
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # Using the stuff method
        retriever=vector_store.as_retriever(),
        verbose=True
    )

    # User input for query
    query = st.text_input("Enter your query", "Can you give me a summary of that story. Please give me some details and it should be a little long")

    # Run the query through the chain
    if st.button("Generate Answer"):
        retrieved_docs = qa_chain.retriever.get_relevant_documents(query)
        formatted_query = {
            "context": " ".join([doc.page_content for doc in retrieved_docs]),
            "question": query
        }
        result = manual_llm_chain(llm, QA_CHAIN_PROMPT, formatted_query)
        st.write("Generated Answer:")
        st.write(result)
