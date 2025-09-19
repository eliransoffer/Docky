# document_processing/vector_store.py
from typing import List, Optional
import os
import shutil
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class VectorStoreManager:
    """Manages Chroma vector store operations"""
    
    def __init__(self, collection_name: str, persist_directory: str, embeddings: GoogleGenerativeAIEmbeddings):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embeddings = embeddings
        self.vector_store: Optional[Chroma] = None
        self.initialized = False
    
    def initialize_store(self) -> Chroma:
        """Initialize or load existing vector store"""
        try:
            self.vector_store = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
            self.initialized = True
            return self.vector_store
        except Exception as e:
            self.initialized = False
            raise ValueError(f"Failed to initialize vector store: {str(e)}")
    
    def is_populated(self) -> bool:
        """Check if vector store already contains documents"""
        if not self.vector_store or not self.initialized:
            return False
        
        try:
            existing_docs = self.vector_store.get()
            return len(existing_docs.get("ids", [])) > 0
        except Exception:
            return False
    
    def add_documents(self, documents: List[Document]) -> int:
        """Add documents to vector store"""
        if not self.vector_store or not self.initialized:
            raise ValueError("Vector store not initialized")
        
        try:
            self.vector_store.add_documents(documents)
            return len(documents)
        except Exception as e:
            raise ValueError(f"Failed to add documents to vector store: {str(e)}")
    
    def get_retriever(self, search_kwargs: Optional[dict] = None):
        """Get retriever for the vector store"""
        if not self.vector_store or not self.initialized:
            raise ValueError("Vector store not initialized")
        
        search_kwargs = search_kwargs or {"k": 6}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
    
    def get_store_info(self) -> dict:
        """Get information about the vector store"""
        if not self.vector_store or not self.initialized:
            return {"initialized": False}
        
        try:
            store_data = self.vector_store.get()
            document_count = len(store_data.get("ids", []))
            
            # Get unique documents and pages
            unique_docs = set()
            unique_pages = set()
            
            metadatas = store_data.get("metadatas", [])
            for metadata in metadatas:
                if metadata:
                    doc_name = metadata.get("document_name")
                    page = metadata.get("page")
                    
                    if doc_name:
                        unique_docs.add(doc_name)
                    if page is not None:
                        unique_pages.add(page)
            
            return {
                "initialized": True,
                "collection_name": self.collection_name,
                "persist_directory": self.persist_directory,
                "document_count": document_count,
                "unique_documents": len(unique_docs),
                "unique_pages": len(unique_pages),
                "documents": list(unique_docs)
            }
        except Exception as e:
            return {
                "initialized": True,
                "error": f"Failed to get store info: {str(e)}"
            }
    
    def search_similar(self, query: str, k: int = 5) -> List[Document]:
        """Search for similar documents"""
        if not self.vector_store or not self.initialized:
            raise ValueError("Vector store not initialized")
        
        try:
            results = self.vector_store.similarity_search(query, k=k)
            return results
        except Exception as e:
            raise ValueError(f"Failed to search vector store: {str(e)}")
    
    def delete_collection(self):
        """Delete the entire collection and clean up all resources"""
        success = False
        
        # Method 1: Try to delete collection through Chroma
        if self.vector_store:
            try:
                self.vector_store.delete_collection()
                print(f"✅ Deleted Chroma collection: {self.collection_name}")
                success = True
            except Exception as e:
                print(f"⚠️ Failed to delete collection via Chroma: {str(e)}")
        
        # Method 2: Try to delete the persist directory directly
        if self.persist_directory and os.path.exists(self.persist_directory):
            try:
                # Get the collection directory path
                collection_path = os.path.join(self.persist_directory, "chroma.sqlite3")
                collection_dir = self.persist_directory
                
                # Remove the entire persist directory
                if os.path.exists(collection_dir):
                    shutil.rmtree(collection_dir)
                    print(f"✅ Deleted persist directory: {collection_dir}")
                    success = True
            except Exception as e:
                print(f"⚠️ Failed to delete persist directory: {str(e)}")
        
        # Method 3: Try alternative cleanup approaches
        if not success:
            print("⚠️ Standard deletion methods failed, attempting alternative cleanup...")
            
            # Try to reset the vector store instance
            try:
                if hasattr(self.vector_store, '_client'):
                    # For newer versions of Chroma
                    client = self.vector_store._client
                    if hasattr(client, 'delete_collection'):
                        client.delete_collection(self.collection_name)
                        success = True
                        print(f"✅ Deleted collection via client: {self.collection_name}")
            except Exception as e:
                print(f"⚠️ Alternative deletion failed: {str(e)}")
        
        # Always reset the manager state
        self._reset_state()
        
        if not success:
            print(f"⚠️ Could not fully delete collection {self.collection_name}, but state has been reset")
    
    def _reset_state(self):
        """Reset the manager's internal state"""
        self.vector_store = None
        self.initialized = False
        print("✅ Vector store manager state reset")
    
    def clear_documents(self):
        """Clear all documents from the collection without deleting the collection"""
        if not self.vector_store or not self.initialized:
            raise ValueError("Vector store not initialized")
        
        try:
            # Get all document IDs
            store_data = self.vector_store.get()
            doc_ids = store_data.get("ids", [])
            
            if doc_ids:
                # Delete all documents
                self.vector_store.delete(ids=doc_ids)
                print(f"✅ Cleared {len(doc_ids)} documents from collection")
            else:
                print("ℹ️ Collection is already empty")
                
        except Exception as e:
            print(f"⚠️ Failed to clear documents: {str(e)}")
            # If clearing fails, try to delete and recreate the collection
            self.delete_collection()
    
    def recreate_collection(self):
        """Recreate the collection from scratch"""
        try:
            # Delete existing collection
            self.delete_collection()
            
            # Reinitialize
            self.initialize_store()
            print(f"✅ Recreated collection: {self.collection_name}")
            
        except Exception as e:
            print(f"❌ Failed to recreate collection: {str(e)}")
            raise