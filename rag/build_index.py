from dataclasses import dataclass
from config import INDEX_DIR
from chunker import Chunk
from faiss import IndexFlatIP

@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    
class VectorStore:
    def __init__(self) -> None:
        pass