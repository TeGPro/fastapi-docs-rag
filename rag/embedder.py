import numpy as np
from sentence_transformers import SentenceTransformer
from rag.config import EMBEDDING_MODEL, EMBEDDING_DIM

class Embedder():
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        self.model = SentenceTransformer(model_name)
    
    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        
        emb = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return emb.astype(np.float32)
    
    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_texts([query])[0]
