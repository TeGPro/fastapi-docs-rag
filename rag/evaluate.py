import json
import time
from pathlib import Path

from rag.config import TOP_K, INDEX_DIR
from rag.embedder import Embedder
from rag.vector_store import VectorStore, SearchResult
from rag.generator import answer_question

def load_eval_set(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
    
def retrieval_hit(item: dict, results: list[SearchResult]) -> bool:
    return any(r.chunk.doc_path == item["expected_doc"] for r in results)

def run_eval(k: int = TOP_K) -> None:
    embedder = Embedder()
    store = VectorStore.load(INDEX_DIR)
    eval_set = load_eval_set(Path("data/eval_questions.json"))

    hits = 0
    counted = 0

    for item in eval_set:
        q = item["question"]
        ans = answer_question(q, embedder, store, k=k)

        if item["expected_doc"] is None:
            refused = "does not cover" in ans.text.lower()
            print(f"[TRAP] {q}\n  refused={refused}\n  answer: {ans.text[:200]}\n")
        else:
            hit = retrieval_hit(item, ans.sources)
            hits += hit
            counted += 1
            srcs = {s.chunk.doc_path for s in ans.sources}
            print(f"[hit={hit}] {q}\n  sources: {srcs}\n  answer: {ans.text[:200]}\n")

        time.sleep(2)

    print(f"\nhit-rate@{k} = {hits}/{counted} = {hits/counted:.2f}")
    
if __name__ == "__main__":
    run_eval()