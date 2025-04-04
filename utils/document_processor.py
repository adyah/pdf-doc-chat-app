"""
Document processor module for handling various document types.
"""

import os
import tempfile
from typing import List, Optional, Dict, Any, Union

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredFileLoader,
    CSVLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class DocumentProcessor:
    """
    Handles document loading, processing, and vector store creation.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        persist_directory: str = "chroma_db"
    ):
        """
        Initialize the document processor.
        
        Args:
            chunk_size: Size of text chunks for splitting
            chunk_overlap: Overlap between text chunks
            embedding_model_name: Name of the Hugging Face model for embeddings
            persist_directory: Directory to store the vector database
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model_name = embedding_model_name
        self.persist_directory = persist_directory
        
        # Initialize the text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len
        )
        
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        # Initialize the embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name
        )
    
    def get_loader_for_file(self, file_path: str) -> Optional[Any]:
        """
        Get the appropriate document loader based on file extension.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Document loader instance or None if unsupported
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return PyPDFLoader(file_path)
        elif file_extension == '.txt':
            return TextLoader(file_path)
        elif file_extension == '.csv':
            return CSVLoader(file_path)
        elif file_extension in ['.docx', '.doc']:
            return Docx2txtLoader(file_path)
        elif file_extension in ['.md', '.html', '.xml', '.json']:
            return UnstructuredFileLoader(file_path)
        else:
            # Try with UnstructuredFileLoader as fallback
            try:
                return UnstructuredFileLoader(file_path)
            except Exception:
                return None
    
    def process_file(self, file_path: str) -> List[Document]:
        """
        Process a single file and return its content as documents.
        
        Args:
            file_path: Path to the file
            
        Returns:
            List of Document objects
            
        Raises:
            ValueError: If file type is unsupported or processing fails
        """
        loader = self.get_loader_for_file(file_path)
        
        if loader is None:
            raise ValueError(f"Unsupported file type: {file_path}")
        
        try:
            return loader.load()
        except Exception as e:
            raise ValueError(f"Error processing file {file_path}: {str(e)}")
    
    def process_documents(
        self,
        file_contents: List[Dict[str, Union[str, bytes]]],
    ) -> Dict[str, Any]:
        """
        Process multiple documents and create a retriever chain.
        
        Args:
            file_contents: List of dictionaries with file names and content
            
        Returns:
            Dictionary with processing result information
            
        Raises:
            ValueError: If no documents were successfully processed
        """
        all_documents = []
        processed_files = 0
        
        for file_info in file_contents:
            file_name = file_info.get("name", "")
            file_content = file_info.get("content", b"")
            
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
                if isinstance(file_content, str):
                    tmp_file.write(file_content.encode('utf-8'))
                else:
                    tmp_file.write(file_content)
                temp_file_path = tmp_file.name
            
            try:
                documents = self.process_file(temp_file_path)
                if documents:
                    # Add source metadata if not present
                    for doc in documents:
                        if "source" not in doc.metadata:
                            doc.metadata["source"] = file_name
                    
                    all_documents.extend(documents)
                    processed_files += 1
            except Exception as e:
                print(f"Error processing {file_name}: {str(e)}")
            finally:
                # Remove the temporary file
                os.remove(temp_file_path)
        
        if not all_documents:
            raise ValueError("No documents were successfully processed")
        
        # Split the documents into chunks
        splits = self.text_splitter.split_documents(all_documents)
        
        # Create vector store
        db = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Create retriever
        retriever = db.as_retriever(search_kwargs={"k": min(len(db), 4)})
        
        return {
            "num_files_processed": processed_files,
            "num_chunks": len(splits),
            "retriever": retriever
        }
