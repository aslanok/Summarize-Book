# Text Summarization and QA Application

This application allows users to upload documents in various formats (PDF, DOCX, TXT, MD) and receive detailed summaries or answers to specific queries about the content. The application leverages advanced AI models and a local instance of Ollama's LLM for natural language processing tasks.

## Features

- **Multiple File Format Support**: Upload and process PDF, DOCX, TXT, and Markdown files.
- **Text Summarization**: Automatically summarize the content of the uploaded document.
- **Question-Answering**: Ask any question related to the document, and the AI provides a precise answer.
- **Local LLM with Ollama**: Utilize the power of local language models using Ollama for high performance and privacy.

## Prerequisites

- **Python 3.8 or later**
- **Virtual Environment**: It is recommended to use a virtual environment to manage dependencies.
- **Ollama**: Ensure you have Ollama installed and set up on your local machine. [Ollama Installation Guide](https://ollama.com).

## Installation

1. **Clone the Repository**
    ```bash
    git clone https://github.com/aslanok/Summarize-Book.git
    cd Summarize-Book
    ```

2. **Create a Virtual Environment**
    It's highly recommended to create a virtual environment to manage dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. **Install Dependencies**
    All required Python packages are listed in the `requirements.txt` file. Install them using pip:
    ```bash
    pip install -r requirements.txt
    ```

4. **Configure Ollama (for Local LLM)**
    Make sure Ollama is properly configured and the required model is downloaded. The application uses the `gemma2:2b` model. You can configure Ollama by following the official [Ollama Documentation](https://ollama.com).

## Usage

1. **Start the Application**
    Run the Streamlit application:
    ```bash
    streamlit run app.py
    ```

2. **Upload a Document**
    - Navigate to the Streamlit interface in your browser.
    - Upload a document (PDF, DOCX, TXT, or MD).
    - The document will be processed, and its content will be split into manageable chunks for analysis.

3. **Ask a Question or Request a Summary**
    - Enter your query in the provided text input field. You can ask for a summary or any specific details about the document.
    - Click the "Generate Answer" button.
    - The application will retrieve relevant sections from the document and generate a concise answer using the local LLM.

## How It Works

- **File Processing**: The application reads and processes different file formats, converting the content into plain text.
- **Text Splitting**: The text is split into smaller chunks using the `RecursiveCharacterTextSplitter` to manage large documents efficiently.
- **Embedding and Indexing**: The text chunks are embedded using the `SentenceTransformer` model and stored in a `ChromaDB` instance.
- **Question-Answering**: The application uses a local LLM (Ollama) combined with a `ChromaDB`-backed retrieval mechanism to answer user queries.
- **Local LLM Integration**: The Ollama model is used to generate accurate and context-aware responses, ensuring privacy and performance by keeping the computation local.


## Support

If you encounter any issues or have questions, please create an issue in the [GitHub repository](https://github.com/aslanok/Summarize-Book).

