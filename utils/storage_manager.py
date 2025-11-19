import os
import json
from pathlib import Path
from datetime import datetime
from config import UPLOAD_DIR, SUMMARY_DIR

class StorageManager:
    """Handles all local file storage operations"""
    
    @staticmethod
    def save_uploaded_file(uploaded_file, file_type="excel"):
        """Save uploaded file to local storage"""
        try:
            file_path = UPLOAD_DIR / uploaded_file.name
            
            # Write file
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Create metadata file
            metadata = {
                "filename": uploaded_file.name,
                "size_bytes": uploaded_file.size,
                "uploaded_at": datetime.now().isoformat(),
                "file_type": file_type,
                "file_path": str(file_path)
            }
            
            meta_file = UPLOAD_DIR / f"{uploaded_file.name}.meta.json"
            with open(meta_file, "w") as f:
                json.dump(metadata, f, indent=2)
            
            return str(file_path)
        
        except Exception as e:
            raise Exception(f"Error saving file: {str(e)}")
    
    @staticmethod
    def get_uploaded_files(file_type="excel"):
        """Get list of all uploaded files"""
        files = []
        
        for file in UPLOAD_DIR.glob("*"):
            # Skip metadata files
            if file.suffix == ".json":
                continue
            
            # Check file type
            if file_type == "excel" and file.suffix not in ['.xlsx', '.xls']:
                continue
            
            # Load metadata
            meta_file = file.parent / f"{file.name}.meta.json"
            metadata = None
            
            if meta_file.exists():
                try:
                    with open(meta_file) as f:
                        metadata = json.load(f)
                except:
                    pass
            
            files.append({
                "path": str(file),
                "name": file.name,
                "meta": metadata
            })
        
        # Sort by upload time (newest first)
        files.sort(
            key=lambda x: x.get("meta", {}).get("uploaded_at", "") if x.get("meta") else "",
            reverse=True
        )
        
        return files
    
    @staticmethod
    def save_summary(document_name, summary_text):
        """Save a generated summary to disk"""
        try:
            summary_path = SUMMARY_DIR / f"{Path(document_name).stem}_summary.txt"
            
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary_text)
            
            # Create metadata
            meta = {
                "original_file": document_name,
                "summary_length": len(summary_text),
                "created_at": datetime.now().isoformat()
            }
            
            meta_file = SUMMARY_DIR / f"{Path(document_name).stem}_summary.meta.json"
            with open(meta_file, "w") as f:
                json.dump(meta, f, indent=2)
            
            return str(summary_path)
        
        except Exception as e:
            raise Exception(f"Error saving summary: {str(e)}")
