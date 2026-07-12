from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.config import INDEX_DIR

e = Embedder()
store = VectorStore.load(INDEX_DIR)   # с диска, не пересобирая

questions = [
    "How do I declare optional query parameters?",
    "How to return a custom status code?",
    "How does dependency injection work?",
    "How to handle file uploads?",
    "How to add CORS?",
]

for q in questions:
    print(f"\n{q}")
    for r in store.search(e.embed_query(q), k=3):
        print(f"  {r.score:.3f}  {r.chunk.doc_path}  |  {r.chunk.heading}")