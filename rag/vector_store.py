from dataclasses import dataclass, asdict
from rag.config import INDEX_DIR, EMBEDDING_DIM
from rag.chunker import Chunk
from faiss import IndexFlatIP, write_index, read_index
from pathlib import Path
import numpy as np
import json


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    
class VectorStore:
    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.index = IndexFlatIP(dim)
        self.chunks: list[Chunk] = []
    
    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        assert len(embeddings) == len(chunks)
        self.index.add(embeddings)
        self.chunks.extend(chunks)
    
    def search(self, query_emb: np.ndarray, k:int = 5) -> list[SearchResult]:
        if self.index.ntotal == 0:
 
            return []     
        q = query_emb.reshape(1, -1)
        scores, positions = self.index.search(q, k)
        
        res = []
        
        for (pos, score) in zip(positions[0], scores[0]):
            if pos == -1:
                continue
            res.append(SearchResult(
                chunk=self.chunks[int(pos)],
                score=float(score)
            ))
        return res
    
    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        write_index(self.index, str(index_dir / "faiss_index"))
        chunks_data = [asdict(c) for c in self.chunks]
        
        with open(index_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
     
    @classmethod            
    def load(cls, index_dir: Path) -> "VectorStore":
        if not (index_dir / "faiss_index").exists():
            raise FileNotFoundError("run scripts/build_index.py first")
        index = read_index(str(index_dir / "faiss_index"))
        with open(index_dir / "chunks.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        chunks = [Chunk(**d) for d in data]

        store = cls()
        store.index = index
        store.chunks = chunks
        return store