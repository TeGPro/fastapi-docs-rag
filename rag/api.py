from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel, Field

from rag.config import INDEX_DIR, TOP_K
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.generator import answer_question

embedder: Embedder | None = None
store: VectorStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global embedder, store
    embedder = Embedder()
    try:
        store = VectorStore.load(INDEX_DIR)
    except FileNotFoundError:
        # падаем на старте с понятным сообщением, а не на первом запросе
        raise RuntimeError("Index not found. Run: python scripts/build_index.py")
    yield


app = FastAPI(lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(default=TOP_K, ge=1, le=20)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]   # уникальные doc_path, порядок релевантности сохранён


def unique_paths(sources) -> list[str]:
    """doc_path без дублей, порядок первого появления."""
    return list(dict.fromkeys(r.chunk.doc_path for r in sources))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    ans = answer_question(req.question, embedder, store, k=req.k)
    return AskResponse(answer=ans.text, sources=unique_paths(ans.sources))