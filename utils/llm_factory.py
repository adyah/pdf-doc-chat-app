"""
Factory module for handling different LLM providers.
"""

import os
from typing import Optional, Dict, Any, List
from enum import Enum
from dotenv import load_dotenv

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.retrievers import BaseRetriever

# Load environment variables
load_dotenv()


class LLMProvider(str, Enum):
    """Enum for supported LLM providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    OLLAMA = "ollama"
    HUGGINGFACE = "huggingface"


class LLMFactory:
    """Factory for creating LLM instances and RAG chains."""
    
    def __init__(self):
        """Initialize the LLM factory."""
        self.provider_mapping = {
            LLMProvider.OPENAI: self._create_openai_llm,
            LLMProvider.ANTHROPIC: self._create_anthropic_llm,
            LLMProvider.GROQ: self._create_groq_llm,
            LLMProvider.OLLAMA: self._create_ollama_llm,
            LLMProvider.HUGGINGFACE: self._create_huggingface_llm,
        }
    
    def _create_openai_llm(self, model_name: str, **kwargs) -> Optional[BaseChatModel]:
        """Create an OpenAI LLM."""
        try:
            from langchain_openai import ChatOpenAI
            
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            return ChatOpenAI(
                model_name=model_name or "gpt-3.5-turbo",
                temperature=kwargs.get("temperature", 0.5),
                api_key=api_key
            )
        except ImportError:
            raise ImportError("langchain_openai is not installed. Install it with 'pip install langchain-openai'")
    
    def _create_anthropic_llm(self, model_name: str, **kwargs) -> Optional[BaseChatModel]:
        """Create an Anthropic LLM."""
        try:
            from langchain_anthropic import ChatAnthropic
            
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set")
            
            return ChatAnthropic(
                model_name=model_name or "claude-3-sonnet-20240229",
                temperature=kwargs.get("temperature", 0.5),
                api_key=api_key
            )
        except ImportError:
            raise ImportError("langchain_anthropic is not installed. Install it with 'pip install langchain-anthropic'")
    
    def _create_groq_llm(self, model_name: str, **kwargs) -> Optional[BaseChatModel]:
        """Create a Groq LLM."""
        try:
            from langchain_groq import ChatGroq
            
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY environment variable is not set")
            
            return ChatGroq(
                model_name=model_name or "llama-3.1-8b-instant",
                temperature=kwargs.get("temperature", 0.5),
                api_key=api_key
            )
        except ImportError:
            raise ImportError("langchain_groq is not installed. Install it with 'pip install langchain-groq'")
    
    def _create_ollama_llm(self, model_name: str, **kwargs) -> Optional[BaseChatModel]:
        """Create an Ollama LLM."""
        try:
            from langchain_ollama import ChatOllama
            
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            
            return ChatOllama(
                model=model_name or "llama3",
                temperature=kwargs.get("temperature", 0.5),
                base_url=base_url
            )
        except ImportError:
            raise ImportError("langchain_ollama is not installed. Install it with 'pip install langchain-ollama'")
    
    def _create_huggingface_llm(self, model_name: str, **kwargs) -> Optional[BaseChatModel]:
        """Create a Hugging Face LLM."""
        try:
            from langchain_huggingface import HuggingFaceEndpoint
            
            api_key = os.getenv("HF_API_KEY")
            if not api_key:
                raise ValueError("HF_API_KEY environment variable is not set")
            
            return HuggingFaceEndpoint(
                repo_id=model_name or "mistralai/Mistral-7B-Instruct-v0.2",
                temperature=kwargs.get("temperature", 0.5),
                huggingfacehub_api_token=api_key
            )
        except ImportError:
            raise ImportError("langchain_huggingface is not installed. Install it with 'pip install langchain-huggingface'")
    
    def get_llm(
        self,
        provider: str,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseChatModel:
        """
        Get an LLM instance from the specified provider.
        
        Args:
            provider: LLM provider name
            model_name: Model name to use
            **kwargs: Additional arguments for the LLM
            
        Returns:
            LLM instance
            
        Raises:
            ValueError: If provider is not supported
        """
        try:
            provider_enum = LLMProvider(provider.lower())
            create_func = self.provider_mapping.get(provider_enum)
            
            if create_func:
                return create_func(model_name, **kwargs)
            else:
                raise ValueError(f"Unsupported LLM provider: {provider}")
        except ValueError:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    def create_rag_chain(
        self,
        retriever: BaseRetriever,
        llm: BaseChatModel,
        prompt_template: Optional[str] = None
    ) -> Any:
        """
        Create a RAG chain using the given retriever and LLM.
        
        Args:
            retriever: Document retriever
            llm: LLM instance
            prompt_template: Optional custom prompt template
            
        Returns:
            RAG chain
        """
        # Define the default prompt template if none is provided
        if not prompt_template:
            prompt_template = """
            You are a helpful AI assistant. Answer the user's question based on the provided context.
            If you don't know the answer, just say "I don't know" or "I don't have enough information to answer that."
            Don't make up answers that aren't supported by the context.
            
            Context:
            {context}
            
            Question:
            {question}
            
            Answer:
            """
        
        # Create the prompt
        prompt = ChatPromptTemplate.from_template(prompt_template)
        
        # Create the RAG chain
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain
