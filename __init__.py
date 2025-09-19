# __init__.py
"""
RAG System with Conversation Memory

A modular RAG (Retrieval-Augmented Generation) system that maintains
conversation history with automatic summarization.
"""

from core import RAGWithMemory
from config import RAGConfig, get_config

__version__ = "1.0.0"
__author__ = "RAG System Team"

__all__ = [
    'RAGWithMemory',
    'RAGConfig', 
    'get_config',
]