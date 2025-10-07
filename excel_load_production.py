import os
import uuid
import json
import time
import hashlib
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

# Configure logging without Unicode emojis
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'MAX_FILE_SIZE_MB': int(os.getenv('MAX_FILE_SIZE_MB', '200')),
    'CHUNK_SIZE': int(os.getenv('CHUNK_SIZE', '100')),
    'CACHE_DIR': os.getenv('CACHE_DIR', r'C:\Users\laksh\OneDrive\Desktop\smart excel database'),
    'SUPPORTED_FORMATS': ['.xlsx', '.xls', '.csv', '.tsv'],
    'UPLOADED_FILES_DIR': os.path.join(r'C:\Users\laksh\OneDrive\Desktop\smart excel database', 'uploaded_files'),
    'PROCESSED_DATA_DIR': os.path.join(r'C:\Users\laksh\OneDrive\Desktop\smart excel database', 'processed_data'),
}

# Import vector store with fallback
try:
    from vector_store import (
        get_simple_vector_store, process_excel_simple, search_simple
    )
    logger.info("Vector store modules imported successfully")
except ImportError:
    logger.warning("Vector store modules not available - will use fallback")
    get_simple_vector_store = None
    process_excel_simple = None
    search_simple = None

def ensure_directories_exist():
    """Ensure required directories exist"""
    directories = [
        CONFIG['CACHE_DIR'],
        CONFIG['UPLOADED_FILES_DIR'], 
        CONFIG['PROCESSED_DATA_DIR']
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.info(f"Directory ensured: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")

# Call directory creation
ensure_directories_exist()

@dataclass
class SimpleProcessingResult:
    """Simplified processing result without session dependencies"""
    success: bool
    processing_id: str
    file_path: str
    start_time: datetime
    end_time: Optional[datetime] = None
    data: Optional[Dict[str, pd.DataFrame]] = None
    chunks: Optional[List[Dict]] = None
    error_message: Optional[str] = None
    quality_score: float = 0.0
    
    @property
    def processing_time(self) -> float:
        """Calculate processing time in seconds"""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

class SimpleDataProcessor:
    """Simplified data processor without session validation"""
    
    def __init__(self):
        logger.info("DataProcessor initialized")
    
    def process_uploaded_file(self, uploaded_file, file_hash: str) -> Dict[str, pd.DataFrame]:
        """Process uploaded file using pandas"""
        file_extension = Path(uploaded_file.name).suffix.lower()
        
        try:
            if file_extension in ['.xlsx', '.xls']:
                # Load all sheets
                data = pd.read_excel(uploaded_file, sheet_name=None, engine="openpyxl")
                # Clean each dataframe
                cleaned_data = {}
                for sheet_name, df in data.items():
                    cleaned_data[sheet_name] = self._clean_dataframe(df, sheet_name)
                logger.info(f"Processed {len(cleaned_data)} sheets")
                return cleaned_data
                
            elif file_extension == '.csv':
                df = pd.read_csv(uploaded_file)
                cleaned_df = self._clean_dataframe(df, "Sheet1")
                logger.info("Processed CSV file")
                return {"Sheet1": cleaned_df}
            else:
                logger.warning(f"Unsupported file format: {file_extension}")
                return {}
                
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return {}
    
    def _clean_dataframe(self, df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        """Clean and normalize dataframe"""
        try:
            # Remove completely empty rows and columns
            original_shape = df.shape
            df = df.dropna(how='all').dropna(axis=1, how='all')
            
            if df.empty:
                logger.warning(f"Sheet '{sheet_name}' is empty after cleaning")
                return df
            
            # Clean column names
            df.columns = [
                str(col).strip().replace('\n', ' ').replace('\r', '')
                for col in df.columns
            ]
            
            # Handle unnamed columns
            df.columns = [
                col if col and not str(col).startswith('Unnamed') else f"Column_{i}"
                for i, col in enumerate(df.columns)
            ]
            
            # Reset index
            df = df.reset_index(drop=True)
            
            logger.debug(f"Cleaned sheet '{sheet_name}': {original_shape} -> {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error cleaning dataframe for sheet '{sheet_name}': {e}")
            return pd.DataFrame()
    
    def create_chunks(self, data: Dict[str, pd.DataFrame], file_path: str, file_hash: str) -> List[Dict]:
        """Create chunks from processed data"""
        chunk_size = CONFIG['CHUNK_SIZE']
        chunks = []
        
        for sheet_name, df in data.items():
            if df.empty:
                continue
            
            # Create row-based chunks
            for i in range(0, len(df), chunk_size):
                end_idx = min(i + chunk_size, len(df))
                chunk_df = df.iloc[i:end_idx]
                
                # Convert chunk to text
                chunk_text = self._dataframe_to_text(chunk_df)
                
                # Create metadata
                metadata = {
                    'chunk_id': str(uuid.uuid4()),
                    'file_path': file_path,
                    'sheet_name': sheet_name,
                    'row_range': f"{i}-{end_idx-1}",
                    'columns': ",".join(str(c) for c in df.columns),
                    'chunk_type': "row_based",
                    'created_at': datetime.now().isoformat(),
                    'file_hash': file_hash,
                    'data_quality_score': self._calculate_chunk_quality(chunk_df)
                }
                
                chunk = {
                    'id': metadata['chunk_id'],
                    'text': chunk_text,
                    'metadata': metadata
                }
                chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks from {len(data)} sheets")
        return chunks
    
    def _dataframe_to_text(self, df: pd.DataFrame) -> str:
        """Convert dataframe to text representation"""
        try:
            lines = []
            lines.append(f"Columns: {', '.join(df.columns)}")
            
            for idx, row in df.iterrows():
                row_items = []
                for col, value in row.items():
                    if pd.notna(value):
                        clean_value = str(value).strip()
                        if clean_value:
                            row_items.append(f"{col}: {clean_value}")
                
                if row_items:
                    lines.append(f"Row {idx}: {'; '.join(row_items)}")
            
            result = "\n".join(lines)
            return result[:5000]  # Limit chunk size
            
        except Exception as e:
            logger.warning(f"Error converting dataframe to text: {e}")
            return f"Data summary: {len(df)} rows, {len(df.columns)} columns"
    
    def _calculate_chunk_quality(self, df: pd.DataFrame) -> float:
        """Calculate quality score for chunk"""
        try:
            if df.empty:
                return 0.0
            
            # Calculate completeness (non-null values)
            completeness = (df.count().sum()) / (df.size)
            
            # Calculate data type diversity
            numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
            text_cols = len(df.select_dtypes(include=['object']).columns)
            diversity = min(1.0, (numeric_cols + text_cols) / len(df.columns))
            
            # Combine metrics
            quality_score = (completeness * 0.7) + (diversity * 0.3)
            return round(quality_score, 3)
            
        except Exception as e:
            logger.warning(f"Quality calculation error: {e}")
            return 0.5

class SimpleExcelLoader:
    """Simplified Excel loader without session management"""
    
    def __init__(self):
        self.data_processor = SimpleDataProcessor()
        self.vector_store = None
        self.processing_history = {}
        
        # Initialize vector store if available
        if get_simple_vector_store:
            try:
                self.vector_store = get_simple_vector_store()
                logger.info("Vector store initialized")
            except Exception as e:
                logger.warning(f"Vector store initialization failed: {e}")
        
        logger.info("SimpleExcelLoader initialized successfully")
    
    def process_file_simple(self, uploaded_file, **options) -> Dict[str, Any]:
        """Process uploaded file without session validation"""
        processing_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        try:
            # Validate file
            self._validate_file(uploaded_file)
            logger.info(f"Processing file: {uploaded_file.name}")
            
            # Calculate file hash
            file_content = uploaded_file.getvalue()
            file_hash = self._calculate_file_hash(file_content)
            
            # Process data
            uploaded_file.seek(0)
            data = self.data_processor.process_uploaded_file(uploaded_file, file_hash)
            
            # Create chunks
            chunks = []
            if data:
                chunks = self.data_processor.create_chunks(data, uploaded_file.name, file_hash)
            
            # Store in vector store if available
            if self.vector_store and chunks:
                try:
                    storage_result = process_excel_simple(data, uploaded_file.name)
                    logger.info(f"Stored {len(chunks)} vectors successfully")
                except Exception as e:
                    logger.error(f"Vector storage failed: {e}")
            
            # Create result
            result = {
                'success': True,
                'processing_id': processing_id,
                'file_path': uploaded_file.name,
                'start_time': start_time,
                'end_time': datetime.now(),
                'data': data,
                'chunks': chunks,
                'quality_score': self._calculate_quality_score(data, chunks)
            }
            
            # Store in history
            self.processing_history[processing_id] = result
            
            logger.info(f"Successfully processed {uploaded_file.name}")
            return result
            
        except Exception as e:
            error_result = {
                'success': False,
                'processing_id': processing_id,
                'file_path': uploaded_file.name,
                'start_time': start_time,
                'end_time': datetime.now(),
                'error_message': str(e),
                'data': {},
                'chunks': []
            }
            
            logger.error(f"Processing failed for {uploaded_file.name}: {e}")
            return error_result
    
    def _validate_file(self, uploaded_file):
        """Validate uploaded file"""
        if not uploaded_file:
            raise ValueError("No file uploaded")
        
        file_name = uploaded_file.name
        file_extension = Path(file_name).suffix.lower()
        
        # Check format
        if file_extension not in CONFIG['SUPPORTED_FORMATS']:
            raise ValueError(f"Unsupported format: {file_extension}")
        
        # Check size
        file_size = len(uploaded_file.getvalue())
        max_size = CONFIG['MAX_FILE_SIZE_MB'] * 1024 * 1024
        
        if file_size > max_size:
            raise ValueError(f"File too large: {file_size/(1024*1024):.2f}MB > {CONFIG['MAX_FILE_SIZE_MB']}MB")
        
        logger.debug(f"File validation passed: {file_name}")
    
    def _calculate_file_hash(self, file_content: bytes) -> str:
        """Calculate SHA256 hash of file content"""
        hasher = hashlib.sha256()
        hasher.update(file_content)
        return hasher.hexdigest()[:16]
    
    def _calculate_quality_score(self, data: Dict[str, pd.DataFrame], chunks: List[Dict]) -> float:
        """Calculate quality score for processing result"""
        try:
            if not data:
                return 0.0
            
            total_score = 0.0
            sheet_count = 0
            
            for sheet_name, df in data.items():
                if df.empty:
                    continue
                
                # Calculate completeness
                total_cells = df.size
                non_null_cells = df.count().sum()
                completeness = non_null_cells / max(total_cells, 1)
                
                # Calculate consistency
                duplicates = df.duplicated().sum()
                consistency = 1.0 - (duplicates / max(len(df), 1))
                
                # Calculate structure quality
                numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
                total_cols = len(df.columns)
                structure_quality = min(1.0, numeric_cols / max(total_cols, 1) + 0.3)
                
                sheet_score = (completeness * 0.4) + (consistency * 0.3) + (structure_quality * 0.3)
                total_score += sheet_score
                sheet_count += 1
            
            base_quality = total_score / max(sheet_count, 1)
            
            # Factor in chunk generation success
            if chunks:
                chunk_success = min(1.0, len(chunks) / max(sheet_count * 5, 1))
                final_score = (base_quality * 0.8) + (chunk_success * 0.2)
            else:
                final_score = base_quality * 0.7
            
            return round(final_score, 3)
            
        except Exception as e:
            logger.warning(f"Quality score calculation failed: {e}")
            return 0.5
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get loader statistics"""
        successful_processes = sum(1 for r in self.processing_history.values() if r.get('success', False))
        
        stats = {
            'total_processes': len(self.processing_history),
            'successful_processes': successful_processes,
            'failed_processes': len(self.processing_history) - successful_processes,
            'success_rate': round((successful_processes / max(len(self.processing_history), 1)) * 100, 2),
            'vector_storage_available': self.vector_store is not None,
            'data_available': successful_processes > 0
        }
        
        if successful_processes > 0:
            total_time = 0
            for result in self.processing_history.values():
                if result.get('success') and result.get('end_time') and result.get('start_time'):
                    time_diff = (result['end_time'] - result['start_time']).total_seconds()
                    total_time += time_diff
            
            stats['average_processing_time'] = round(total_time / successful_processes, 2)
        else:
            stats['average_processing_time'] = 0.0
        
        return stats
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary of available data"""
        try:
            summary = {
                'files_processed': [],
                'total_chunks': 0,
                'sheets_available': [],
                'last_updated': None
            }
            
            # Get data from processing history
            for result in self.processing_history.values():
                if result.get('success') and result.get('data'):
                    file_name = Path(result['file_path']).name
                    if file_name not in summary['files_processed']:
                        summary['files_processed'].append(file_name)
                    
                    if result['data']:
                        for sheet_name in result['data'].keys():
                            if sheet_name not in summary['sheets_available']:
                                summary['sheets_available'].append(sheet_name)
                    
                    if result.get('chunks'):
                        summary['total_chunks'] += len(result['chunks'])
            
            # Set last updated
            if self.processing_history:
                latest_time = max(
                    (r.get('start_time') for r in self.processing_history.values() 
                     if r.get('start_time')), 
                    default=None
                )
                if latest_time:
                    summary['last_updated'] = latest_time.isoformat()
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting data summary: {e}")
            return {
                'files_processed': [],
                'total_chunks': 0,
                'sheets_available': [],
                'error': str(e)
            }

# Global instance management
_simple_loader_instance = None

def get_simple_loader() -> SimpleExcelLoader:
    """Get singleton simple loader instance"""
    global _simple_loader_instance
    if _simple_loader_instance is None:
        _simple_loader_instance = SimpleExcelLoader()
    return _simple_loader_instance

# Compatibility exports
__all__ = [
    'SimpleExcelLoader',
    'SimpleProcessingResult', 
    'SimpleDataProcessor',
    'get_simple_loader'
]

if __name__ == "__main__":
    # Test the simplified loader
    try:
        loader = get_simple_loader()
        stats = loader.get_statistics()
        print("SimpleExcelLoader Test Results:")
        print("=" * 50)
        for key, value in stats.items():
            print(f"{key}: {value}")
        print("\nSimpleExcelLoader test completed successfully!")
    except Exception as e:
        print(f"Test failed: {e}")
        logger.error(f"Test error: {e}")