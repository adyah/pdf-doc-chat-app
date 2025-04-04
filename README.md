# PDF and Document Chat App

A modern application for chatting with PDF documents and text files using RAG (Retrieval Augmented Generation).

## Features

- Upload and process PDF documents, text files, and other document formats
- Chat with your documents using natural language queries
- Advanced document chunking and retrieval
- Support for multiple LLM providers
- Clean, modern UI with Streamlit

## Getting Started

### Prerequisites

- Python 3.9+
- Required libraries (see requirements.txt)

### Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file with your API keys (if using API-based LLMs)
4. Run the application:
```bash
streamlit run app.py
```

## How It Works

This application uses Retrieval Augmented Generation (RAG) to enable conversational interaction with your documents:

1. Documents are processed and split into chunks
2. Text is converted to vector embeddings and stored in a vector database
3. When you ask a question, the system retrieves the most relevant chunks
4. These chunks are provided as context to the LLM to generate an accurate response

## Technologies Used

- LangChain for orchestration
- Streamlit for the UI
- ChromaDB for vector storage
- Hugging Face models for embeddings
- Support for various LLM providers