from pathlib import Path

def ensure_log_directory(log_file: str) -> Path:
    path = Path(log_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
