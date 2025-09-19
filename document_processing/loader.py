# document_processing/loader.py
import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentLoader:
    """Handles document loading and processing"""
    
    def __init__(self, chunk_size: int = 2000, chunk_overlap: int = 400):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
        )
    
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load PDF document and return raw documents"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            return documents
        except Exception as e:
            raise ValueError(f"Failed to load PDF '{pdf_path}': {str(e)}")
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        try:
            splits = self.text_splitter.split_documents(documents)
            return splits
        except Exception as e:
            raise ValueError(f"Failed to split documents: {str(e)}")
    
    def add_metadata(self, splits: List[Document], pdf_path: str) -> List[Document]:
        """Add enhanced metadata to document splits"""
        document_name = os.path.basename(pdf_path)
        
        for i, split in enumerate(splits):
            split.metadata.update({
                'chunk_id': i,
                'document_name': document_name,
                'chunk_size': len(split.page_content),
                'source_type': 'pdf',
                'source_path': pdf_path
            })
        
        return splits
    
    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Complete PDF processing pipeline"""
        print(f"Loading PDF: {pdf_path}")
        documents = self.load_pdf(pdf_path)
        print(f"Loaded {len(documents)} pages")
        
        print("Splitting documents into chunks...")
        splits = self.split_documents(documents)
        print(f"Created {len(splits)} chunks")
        
        print("Adding metadata...")
        splits_with_metadata = self.add_metadata(splits, pdf_path)
        
        return splits_with_metadata
    
    def get_processing_stats(self, splits: List[Document]) -> dict:
        """Get statistics about processed documents"""
        if not splits:
            return {'total_chunks': 0, 'total_characters': 0, 'avg_chunk_size': 0}
        
        total_chars = sum(len(split.page_content) for split in splits)
        avg_chunk_size = total_chars / len(splits) if splits else 0
        
        # Get page distribution
        pages = set()
        for split in splits:
            if 'page' in split.metadata:
                pages.add(split.metadata['page'])
        
        return {
            'total_chunks': len(splits),
            'total_characters': total_chars,
            'avg_chunk_size': round(avg_chunk_size, 2),
            'unique_pages': len(pages),
            'chunk_size_config': self.chunk_size,
            'chunk_overlap_config': self.chunk_overlap
        }