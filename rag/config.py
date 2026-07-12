from pathlib import Path

root_folder = Path.cwd()

DOCS_ROOT = root_folder / "data"
INDEX_DIR = root_folder / "data" / "index"
EXCLUDE_FILES = {"release-notes.md", "newsletter.md", "management-tasks.md"}

CHUNK_SIZE = 180
CHUNK_OVERLAP = 40

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 5