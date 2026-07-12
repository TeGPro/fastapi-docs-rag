import time
from rag.config import DOCS_ROOT, CHUNK_SIZE, CHUNK_OVERLAP, INDEX_DIR
from rag.data_loader import load_documents
from rag.chunker import chunk_documents
from rag.embedder import Embedder
from rag.vector_store import VectorStore

t = time.perf_counter()

docs = load_documents(DOCS_ROOT)
chunks = chunk_documents(docs, CHUNK_SIZE, CHUNK_OVERLAP)

embedder= Embedder()
embeddings = embedder.embed_texts([c.text for c in chunks])

store = VectorStore()
store.add(embeddings, chunks)
store.save(INDEX_DIR)

elapsed = time.perf_counter() - t
print(f"documents:   {len(docs)}")
print(f"chunks:      {len(chunks)}")
print(f"index size:  {store.index.ntotal} vectors")
print(f"time:        {elapsed:.1f}s")