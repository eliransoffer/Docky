# core/rag_system.py
from typing import Dict, List, Any, Optional
import shutil
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from config.settings import RAGConfig
from memory import ConversationManager
from document_processing import DocumentLoader, VectorStoreManager
from chains import RAGChainBuilder

class RAGWithMemory:
    """Main RAG system with conversation memory"""
    
    def __init__(self, pdf_path: str, config: RAGConfig):
        self.pdf_path = pdf_path
        self.config = config
        
        # Initialize components
        self.embeddings = GoogleGenerativeAIEmbeddings(model=config.embedding_model)
        self.llm = ChatGoogleGenerativeAI(model=config.llm_model)
        
        self.document_loader = DocumentLoader(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap
        )
        
        self.vector_store_manager = VectorStoreManager(
            collection_name=config.collection_name,
            persist_directory=config.persist_directory,
            embeddings=self.embeddings
        )
        
        self.conversation_manager = ConversationManager(
            llm=self.llm,
            max_tokens=config.memory_tokens,
            max_recent_exchanges=config.max_recent_exchanges
        )
        
        self.chain_builder = RAGChainBuilder(self.llm)
        
        # Chain will be set during setup
        self.rag_chain = None
    
    def load_and_process_documents(self) -> str:
        """Load PDF and create/load vector store"""
        # Initialize vector store
        self.vector_store_manager.initialize_store()
        
        # Check if already populated
        if self.vector_store_manager.is_populated():
            store_info = self.vector_store_manager.get_store_info()
            return f"Loaded existing collection with {store_info['document_count']} chunks"
        
        # Process new document
        print("Processing new document...")
        splits = self.document_loader.process_pdf(self.pdf_path)
        
        # Add to vector store
        added_count = self.vector_store_manager.add_documents(splits)
        
        # Get processing stats
        stats = self.document_loader.get_processing_stats(splits)
        
        return f"Created new collection with {added_count} chunks (avg size: {stats['avg_chunk_size']} chars)"
    
    def setup_chain(self):
        """Setup RAG chain with memory support"""
        retriever = self.vector_store_manager.get_retriever(
            search_kwargs={"k": self.config.retrieval_k}
        )
        
        self.rag_chain = self.chain_builder.create_rag_chain(
            retriever=retriever,
            use_memory=True
        )
    
    def ask_with_memory(self, question: str) -> Dict[str, Any]:
        """Ask question with conversation memory"""
        if not self.rag_chain:
            raise ValueError("Chain not setup. Call setup_chain() first.")
        
        # Get conversation context
        conversation_context = self.conversation_manager.get_context_for_prompt()
        
        # Prepare input for the chain
        chain_input = {
            "input": question,
            "conversation_context": conversation_context
        }
        
        try:
            # Get response from chain
            response = self.rag_chain.invoke(chain_input)
            
            # Extract and process sources
            sources = self._extract_sources(response.get("context", []))
            answer = response.get("answer", "No answer found.")
            
            # Add to conversation memory
            self.conversation_manager.add_exchange(question, answer, sources)
            
            return {
                'answer': answer,
                'sources': sources,
                'question': question,
                'conversation_stats': self.conversation_manager.get_conversation_stats()
            }
            
        except Exception as e:
            raise ValueError(f"Failed to process question: {str(e)}")
    
    def ask_without_memory(self, question: str) -> Dict[str, Any]:
        """Ask question without using conversation memory"""
        if not self.rag_chain:
            raise ValueError("Chain not setup. Call setup_chain() first.")
        
        # Create a basic chain without memory
        retriever = self.vector_store_manager.get_retriever(
            search_kwargs={"k": self.config.retrieval_k}
        )
        basic_chain = self.chain_builder.create_rag_chain(
            retriever=retriever,
            use_memory=False
        )
        
        try:
            response = basic_chain.invoke({"input": question})
            sources = self._extract_sources(response.get("context", []))
            answer = response.get("answer", "No answer found.")
            
            return {
                'answer': answer,
                'sources': sources,
                'question': question
            }
            
        except Exception as e:
            raise ValueError(f"Failed to process question: {str(e)}")
    
    def _extract_sources(self, context_docs: List) -> List[Dict[str, Any]]:
        """Extract source information from context documents"""
        sources = []
        
        for doc in context_docs:
            source_info = {
                'page': doc.metadata.get('page', 'Unknown'),
                'document': doc.metadata.get('document_name', self.pdf_path),
                'chunk_id': doc.metadata.get('chunk_id', 'Unknown'),
                'content_preview': self._create_content_preview(doc.page_content)
            }
            sources.append(source_info)
        
        return sources
    
    def _create_content_preview(self, content: str, max_length: int = 200) -> str:
        """Create a preview of document content"""
        if len(content) <= max_length:
            return content
        return content[:max_length] + "..."
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return {
            'config': {
                'pdf_path': self.pdf_path,
                'memory_tokens': self.config.memory_tokens,
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'retrieval_k': self.config.retrieval_k
            },
            'vector_store': self.vector_store_manager.get_store_info(),
            'conversation': self.conversation_manager.get_conversation_stats(),
            'status': 'ready' if self.rag_chain else 'not_setup'
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get conversation summary and recent history"""
        return self.conversation_manager.get_summary_info()
    
    def clear_conversation_history(self):
        """Clear conversation history (keep vector store)"""
        self.conversation_manager = ConversationManager(
            llm=self.llm,
            max_tokens=self.config.memory_tokens,
            max_recent_exchanges=self.config.max_recent_exchanges
        )
        print("✅ Conversation history cleared")
    
    def clear_vector_store(self):
        """Clear the vector store completely"""
        try:
            # Use the delete_collection method from VectorStoreManager
            if self.vector_store_manager and hasattr(self.vector_store_manager, 'delete_collection'):
                self.vector_store_manager.delete_collection()
                print("✅ Vector store collection deleted")
            
            # Fallback: If using Chroma, delete the persist directory
            elif self.config.persist_directory and os.path.exists(self.config.persist_directory):
                shutil.rmtree(self.config.persist_directory)
                print(f"✅ Deleted persist directory: {self.config.persist_directory}")
            
            # Reset the vector store manager with fresh instance
            self.vector_store_manager = VectorStoreManager(
                collection_name=self.config.collection_name,
                persist_directory=self.config.persist_directory,
                embeddings=self.embeddings
            )
            
            # Clear the chain as it's now invalid
            self.rag_chain = None
            print("✅ Vector store manager reset")
            
        except Exception as e:
            print(f"⚠️ Error clearing vector store: {str(e)}")
            # Force reset even if deletion fails
            try:
                self.vector_store_manager = VectorStoreManager(
                    collection_name=self.config.collection_name,
                    persist_directory=self.config.persist_directory,
                    embeddings=self.embeddings
                )
                self.rag_chain = None
                print("✅ Force reset completed")
            except Exception as reset_error:
                print(f"❌ Critical error during force reset: {str(reset_error)}")
                raise
    
    def cleanup(self):
        """Complete cleanup of all resources"""
        try:
            print("🧹 Starting complete cleanup...")
            
            # Clear conversation history
            self.clear_conversation_history()
            
            # Clear vector store
            self.clear_vector_store()
            
            # Reset chain
            self.rag_chain = None
            
            print("✅ Complete cleanup finished")
            
        except Exception as e:
            print(f"⚠️ Error during cleanup: {str(e)}")
    
    def reset_for_new_document(self, new_pdf_path: str):
        """Reset system for a new document"""
        try:
            print(f"🔄 Resetting system for new document: {new_pdf_path}")
            
            # Clear everything
            self.cleanup()
            
            # Update PDF path
            self.pdf_path = new_pdf_path
            
            # Recreate components with fresh state
            self.document_loader = DocumentLoader(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap
            )
            
            self.vector_store_manager = VectorStoreManager(
                collection_name=self.config.collection_name,
                persist_directory=self.config.persist_directory,
                embeddings=self.embeddings
            )
            
            self.conversation_manager = ConversationManager(
                llm=self.llm,
                max_tokens=self.config.memory_tokens,
                max_recent_exchanges=self.config.max_recent_exchanges
            )
            
            print("✅ System reset complete, ready for new document")
            
        except Exception as e:
            print(f"⚠️ Error during reset: {str(e)}")
            raise
    
    def is_ready(self) -> bool:
        """Check if the system is ready to process questions"""
        return (
            self.rag_chain is not None and
            self.vector_store_manager is not None and
            self.vector_store_manager.is_populated()
        )