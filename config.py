import os
from pathlib import Path

# ============ DIRECTORY SETUP ============
BASE_DIR = Path(__file__).parent
UPLOAD_DIR = BASE_DIR / "uploaded_files"
VECTOR_DB_DIR = BASE_DIR / "vector_db"
SUMMARY_DIR = BASE_DIR / "summaries"

# Create directories if they don't exist
for dir_path in [UPLOAD_DIR, VECTOR_DB_DIR, SUMMARY_DIR]:
    dir_path.mkdir(exist_ok=True)

# ============ OLLAMA CONFIGURATION ============
OLLAMA_HOST = "http://localhost:11434"

# ============ MODEL CONFIGURATION ============
LLM_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"

# ============ CHROMA DB CONFIGURATION ============
CHROMA_PERSIST_DIR = str(VECTOR_DB_DIR)

# ============ TEXT PROCESSING CONFIGURATION ============
TEXT_CHUNK_SIZE = 300
TEXT_CHUNK_OVERLAP = 50
MAX_RETRIEVAL_RESULTS = 3
# ============ REPORT CONFIGURATION ============
REPORT_DIR = BASE_DIR / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# Report Templates
REPORT_TEMPLATES = {
    "executive": "Executive Summary Report",
    "technical": "Technical Analysis Report",
    "business": "Business Intelligence Report"
}
