# Simplified Vector Store - No Sessions
import os
import json
import uuid
import logging
import hashlib
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

# LangChain imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# ChromaDB
import chromadb

# BM25 for hybrid search
from rank_bm25 import BM25Okapi
import pandas as pd
import numpy as np

# Configure logging without Unicode emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

@dataclass
class SimpleSearchResult:
    score: float
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    retrieval_method: str

class SimpleExcelTableChunker:
    """Simplified table chunker without session dependencies"""
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_excel_data(self, data: Dict[str, pd.DataFrame], file_path: str) -> List[Document]:
        """Create table-aware chunks"""
        documents = []
        
        for sheet_name, df in data.items():
            if df.empty:
                continue
            
            # Convert to markdown table format
            markdown_table = self._df_to_markdown(df)
            chunks = self._split_table_chunks(df, markdown_table, sheet_name, file_path)
            documents.extend(chunks)
        
        return documents
    
    def _df_to_markdown(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to markdown"""
        if df.empty:
            return ""
        
        summary = f"Sheet contains {len(df)} rows and {len(df.columns)} columns.\n"
        summary += f"Columns: {', '.join(str(col) for col in df.columns)}\n\n"
        
        # Create header row
        headers = "| " + " | ".join(str(col) for col in df.columns) + " |"
        separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"
        
        # Create data rows
        rows = []
        for idx, (_, row) in enumerate(df.iterrows()):
            if idx >= 20:  # Limit rows
                rows.append(f"| ... | ... ({len(df) - idx} more rows) |")
                break
            
            row_values = []
            for val in row:
                if pd.isna(val):
                    row_values.append("")
                else:
                    clean_val = str(val).replace("|", "\\|").replace("\n", " ")
                    if len(clean_val) > 50:
                        clean_val = clean_val[:47] + "..."
                    row_values.append(clean_val)
            
            row_str = "| " + " | ".join(row_values) + " |"
            rows.append(row_str)
        
        return summary + "\n".join([headers, separator] + rows)
    
    def _split_table_chunks(self, df: pd.DataFrame, markdown_table: str,
                           sheet_name: str, file_path: str) -> List[Document]:
        """Split table into chunks"""
        documents = []
        
        if len(markdown_table) <= self.chunk_size:
            # Small table - single chunk
            metadata = {
                "file_path": file_path,
                "sheet_name": sheet_name,
                "chunk_type": "table_complete",
                "row_start": 0,
                "row_end": len(df) - 1,
                "columns": ",".join(str(c) for c in df.columns),
                "total_rows": len(df),
                "chunk_id": str(uuid.uuid4()),
                "has_numeric_data": len(df.select_dtypes(include=[np.number]).columns) > 0,
                "column_count": len(df.columns)
            }
            
            documents.append(Document(
                page_content=markdown_table,
                metadata=metadata
            ))
        
        else:
            # Large table - chunk it
            rows_per_chunk = max(3, self.chunk_size // 200)  # Estimate
            header_context = f"Sheet: {sheet_name} | Columns: {', '.join(df.columns)}\n"
            
            for i in range(0, len(df), max(1, rows_per_chunk - self.overlap)):
                end_idx = min(i + rows_per_chunk, len(df))
                chunk_df = df.iloc[i:end_idx]
                
                chunk_markdown = header_context + self._df_to_markdown(chunk_df)
                
                metadata = {
                    "file_path": file_path,
                    "sheet_name": sheet_name,
                    "chunk_type": "table_chunk",
                    "row_start": i,
                    "row_end": end_idx - 1,
                    "columns": ",".join(str(c) for c in df.columns),
                    "total_rows": len(df),
                    "chunk_index": i // max(1, rows_per_chunk - self.overlap),
                    "chunk_id": str(uuid.uuid4()),
                    "has_numeric_data": len(chunk_df.select_dtypes(include=[np.number]).columns) > 0,
                    "column_count": len(df.columns)
                }
                
                documents.append(Document(
                    page_content=chunk_markdown,
                    metadata=metadata
                ))
                
                if end_idx >= len(df):
                    break
        
        return documents

class SimpleChromaVectorStore:
    """Simplified ChromaDB without session management"""
    
    def __init__(self,
                 persist_directory: str = None,
                 collection_name: str = "simple_excel_vectors",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIR",
            r"C:\Users\laksh\OneDrive\Desktop\smart_excel_chroma_vdb"
        )
        
        self.collection_name = collection_name
        self.lock = threading.RLock()
        
        # Ensure directory exists
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Initialize chunker
        self.table_chunker = SimpleExcelTableChunker(chunk_size=512, overlap=50)
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=self.persist_directory)
        self.vector_store = Chroma(
            client=self.client,
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        # Document cache
        self.documents_cache = []
        self._load_existing_documents()
        
        logger.info("Simple ChromaDB Vector Store initialized")
    
    def _load_existing_documents(self):
        """Load existing documents from ChromaDB"""
        try:
            collection = self.client.get_collection(self.collection_name)
            all_data = collection.get(include=["documents", "metadatas"])
            
            if all_data["documents"]:
                for doc, metadata in zip(all_data["documents"], all_data["metadatas"]):
                    self.documents_cache.append(Document(
                        page_content=doc,
                        metadata=metadata or {}
                    ))
                
                logger.info(f"Loaded {len(self.documents_cache)} existing documents")
        except Exception as e:
            logger.info(f"No existing collection found: {e}")
    
    def process_excel_file(self, data: Dict[str, pd.DataFrame], file_path: str) -> Dict[str, Any]:
        """Process Excel file without session validation"""
        try:
            # Create table-aware chunks
            table_documents = self.table_chunker.chunk_excel_data(data, file_path)
            
            # Add metadata
            for doc in table_documents:
                doc.metadata.update({
                    "file_hash": hashlib.sha256(str(data).encode()).hexdigest()[:16],
                    "created_at": datetime.now().isoformat(),
                    "processing_method": "simple_table_chunking"
                })
            
            # Store in vector database
            if table_documents:
                self.vector_store.add_documents(table_documents)
                self.documents_cache.extend(table_documents)
                self.vector_store.persist()
                
                logger.info(f"Processed {len(table_documents)} chunks from {file_path}")
            
            return {
                "success": True,
                "chunks_created": len(table_documents),
                "file_path": file_path,
                "method": "simple_table_chunking"
            }
        
        except Exception as e:
            logger.error(f"Error processing Excel file: {e}")
            return {
                "success": False,
                "error": str(e),
                "chunks_created": 0
            }
    
    def search(self, query: str, top_k: int = 10) -> List[SimpleSearchResult]:
        """Simple search without session validation"""
        try:
            # Use vector similarity search
            docs = self.vector_store.similarity_search_with_score(query, k=top_k)
            
            results = []
            for doc, score in docs:
                result = SimpleSearchResult(
                    score=float(score),
                    content=doc.page_content,
                    metadata=doc.metadata,
                    chunk_id=doc.metadata.get('chunk_id', str(uuid.uuid4())),
                    retrieval_method="dense"
                )
                results.append(result)
            
            logger.info(f"Search returned {len(results)} results for: {query[:50]}")
            return results
        
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        with self.lock:
            return {
                "total_documents": len(self.documents_cache),
                "embeddings_available": self.embeddings is not None,
                "vector_store_healthy": True,
                "persist_directory": self.persist_directory,
                "collection_name": self.collection_name
            }
    
    def load_all_chunks(self) -> Dict[str, Any]:
        """Load all chunks grouped by file"""
        try:
            files_data = {}
            for doc in self.documents_cache:
                file_path = doc.metadata.get("file_path", "unknown_file")
                if file_path not in files_data:
                    files_data[file_path] = {"chunks": [], "metadata": doc.metadata}
                
                files_data[file_path]["chunks"].append({
                    "text": doc.page_content,
                    "metadata": doc.metadata
                })
            
            logger.info(f"Loaded chunks for {len(files_data)} files")
            return files_data
        
        except Exception as e:
            logger.error(f"Error loading chunks: {e}")
            return {}

# Global instance management
_simple_vector_store = None
_store_lock = threading.Lock()

def get_simple_vector_store() -> SimpleChromaVectorStore:
    """Get singleton simple vector store"""
    global _simple_vector_store
    with _store_lock:
        if _simple_vector_store is None:
            _simple_vector_store = SimpleChromaVectorStore()
        return _simple_vector_store

# Simplified API functions
def process_excel_simple(data: Dict[str, pd.DataFrame], file_path: str) -> Dict[str, Any]:
    """Process Excel file without sessions"""
    return get_simple_vector_store().process_excel_file(data, file_path)

def search_simple(query: str, top_k: int = 10) -> List[SimpleSearchResult]:
    """Simple search without sessions"""
    return get_simple_vector_store().search(query, top_k)

def search_enhanced(query: str, session_id: str = None, top_k: int = 10, filters: Optional[Dict] = None) -> List[SimpleSearchResult]:
    """Enhanced search (compatibility function)"""
    return search_simple(query, top_k)

def load_all_chunks_from_chroma():
    """Load all chunks from ChromaDB"""
    return get_simple_vector_store().load_all_chunks()

def get_system_statistics():
    """Get system statistics"""
    return get_simple_vector_store().get_statistics()

# Export functions and classes
__all__ = [
    'SimpleChromaVectorStore',
    'SimpleSearchResult', 
    'SimpleExcelTableChunker',
    'get_simple_vector_store',
    'process_excel_simple',
    'search_simple',
    'search_enhanced',
    'load_all_chunks_from_chroma',
    'get_system_statistics'
]
