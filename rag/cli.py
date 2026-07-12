import argparse

from rag.config import INDEX_DIR, TOP_K
from rag.embedder import Embedder
from rag.vector_store import VectorStore, SearchResult
from rag.generator import answer_question


def format_sources(sources: list[SearchResult]) -> list[tuple[str, float]]:
    seen: dict[str, float] = {}
    for r in sources:
        path = r.chunk.doc_path
        if path not in seen:
            seen[path] = r.score
    return list(seen.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the FastAPI docs")
    parser.add_argument("question", help="your question")
    parser.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args()

    embedder = Embedder()
    try:
        store = VectorStore.load(INDEX_DIR)
    except FileNotFoundError:
        print("Index not found. Run: python scripts/build_index.py")
        raise SystemExit(1)

    ans = answer_question(args.question, embedder, store, k=args.k)

    print(ans.text)
    print("\nSources:")
    for path, score in format_sources(ans.sources):
        print(f"  {path} (score={score:.3f})")


if __name__ == "__main__":
    main()