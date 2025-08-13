"""
Configuration file for RAG system
Store API keys and other configuration here
"""

import os
from pathlib import Path

# OpenAI Configuration
#OPENAI_API_KEY = ""

# Dataset Configuration
DEFAULT_DATASET_PATH = "cranfield/"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_TOP_K = 5

# LLM Configuration
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_OPENAI_MODEL = "gpt-3.5-turbo"

# Server Configuration
DEFAULT_PORT = 7860
DEFAULT_HOST = "0.0.0.0"

# Retriever Configuration
DEFAULT_RETRIEVER_TYPE = "auto"

def get_openai_api_key():
    """Get OpenAI API key from environment or config"""
    return os.getenv("OPENAI_API_KEY", OPENAI_API_KEY)

def get_config():
    """Get complete configuration"""
    return {
        "openai_api_key": get_openai_api_key(),
        "dataset_path": DEFAULT_DATASET_PATH,
        "chunk_size": DEFAULT_CHUNK_SIZE,
        "top_k": DEFAULT_TOP_K,
        "llm_provider": DEFAULT_LLM_PROVIDER,
        "openai_model": DEFAULT_OPENAI_MODEL,
        "port": DEFAULT_PORT,
        "host": DEFAULT_HOST,
        "retriever_type": DEFAULT_RETRIEVER_TYPE
    }

# Set environment variable if not already set
if not os.getenv("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    print("OpenAI API key set from config file")
else:
    print("OpenAI API key found in environment variables")
