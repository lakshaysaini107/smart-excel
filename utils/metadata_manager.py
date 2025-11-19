import json
from pathlib import Path
from datetime import datetime

from config import UPLOAD_DIR


class MetadataManager:
    """Persist and restore information about uploaded files (path, sheet name, etc.)."""

    METADATA_FILE = UPLOAD_DIR / "file_metadata.json"

    @classmethod
    def load_metadata(cls) -> dict:
        """Load all file metadata from disk."""
        if cls.METADATA_FILE.exists():
            try:
                with open(cls.METADATA_FILE, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                # Corrupted metadata; start fresh
                return {}
        return {}

    @classmethod
    def save_file_metadata(cls, file_path: str, sheet_name: str | None = None) -> None:
        """Save or update metadata for a given file path."""
        metadata = cls.load_metadata()

        file_key = Path(file_path).name
        metadata[file_key] = {
            "path": file_path,
            "sheet_name": sheet_name,
            "last_accessed": datetime.now().isoformat(),
            "file_type": Path(file_path).suffix,
        }

        cls.METADATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(cls.METADATA_FILE, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def get_last_file(cls) -> dict | None:
        """Get the most recently accessed file metadata, if any."""
        metadata = cls.load_metadata()
        if not metadata:
            return None

        sorted_files = sorted(
            metadata.items(),
            key=lambda x: x[1].get("last_accessed", ""),
            reverse=True,
        )
        return sorted_files[0][1] if sorted_files else None


