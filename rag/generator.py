from __future__ import annotations
from groq import Groq
from rag.config import LLM_MODEL, TOP_K
from rag.vector_store import SearchResult
from rag.embedder import Embedder
from rag.vector_store import VectorStore
from dataclasses import dataclass
from dotenv import load_dotenv
import os

@dataclass
class Answer:
    text: str
    sources: list[SearchResult]

SYSTEM_PROMPT = (
    "You are a documentation assistant. "
    "Answer the user's question using ONLY the numbered context passages provided below. "
    "Do not use any prior knowledge or information outside the context.\n\n"
    "Cite the passages you rely on by their number in square brackets, e.g. [1], [2].\n\n"
    "If the context does not contain the answer, respond with exactly: "
    '"The provided documentation does not cover this." '
    "Do not attempt to guess or fill in gaps.\n\n"
    "Keep answers concise. You may include code examples if they appear in the context."
)

_client: Groq | None = None

def _get_client() -> Groq:
    global _client
    if _client is None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to .env or run: export GROQ_API_KEY=..."
            )
        _client = Groq(api_key=key)
        
    return _client


def format_context(results: list[SearchResult]) -> str:
    blocks = []
    for i, result in enumerate(results, start=1):
        chunk = result.chunk
        block = f"[{i}] source: {chunk.doc_path} — {chunk.heading}\n{chunk.text}\n---"
        blocks.append(block)
    return "\n".join(blocks)


def build_messages(question: str, results: list[SearchResult]) -> list[dict]:
    context = format_context(results)
    user_content = (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

def generate(messages: list[dict], model: str = LLM_MODEL) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=model, messages=messages, temperature=0  # type: ignore[arg-type]
    )
    content = resp.choices[0].message.content
    if content is None:
        raise RuntimeError("Groq returned empty content")
    return content

def answer_question(question:str, embedder: Embedder, store: VectorStore, k: int = TOP_K) -> Answer:
    query_emb = embedder.embed_query(question)
    results = store.search(query_emb,k=k)
    
    if not results:
        return Answer(
            text="The provided documentation does not cover this.",
            sources=[]
        )
    
    messages = build_messages(question, results)
    text = generate(messages)
    
    return Answer(
        text=text,
        sources=results
    )