# 04 — Векторное хранилище и поиск

**Время: ~2 часа** · Файлы: `rag/vector_store.py`, `scripts/build_index.py`

## Цель модуля

Главный чекпоинт дня 1: работающий semantic search. На выходе — класс `VectorStore` (FAISS-индекс + метаданные чанков, save/load) и скрипт `build_index.py`, собирающий весь офлайн-пайплайн: документы → чанки → векторы → индекс на диске. После этого модуля ты вводишь вопрос и получаешь top-k релевантных кусков документации.

## Теория-минимум

FAISS хранит матрицу векторов и быстро ищет ближайшие. Нам хватит `IndexFlatIP` — честный полный перебор по inner product: на наших ~2000 векторах это доли миллисекунды, а поскольку векторы нормализованы (модуль 03), inner product == cosine similarity. Приближённые индексы (IVF, HNSW) нужны от миллионов векторов — не наш случай (и это тоже ответ на собеседовании).

Важно понять: **FAISS хранит только числа**. Он возвращает позиции ближайших векторов, а «какой это был текст и из какого файла» — твоя забота: храним `list[Chunk]` параллельно индексу, позиция вектора == индекс в списке.

Углубиться: [FAISS wiki: Getting started](https://github.com/facebookresearch/faiss/wiki/Getting-started).

## Что реализовать

В `rag/config.py`: `INDEX_DIR = Path("data/index")`.

```python
@dataclass
class SearchResult:
    chunk: Chunk
    score: float   # cosine similarity, ~[0..1]

class VectorStore:
    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        # faiss.IndexFlatIP(dim) + пустой список чанков

    def add(self, embeddings: np.ndarray, chunks: list[Chunk]) -> None:
        # assert len(embeddings) == len(chunks); index.add + расширить список

    def search(self, query_emb: np.ndarray, k: int = 5) -> list[SearchResult]:
        # FAISS ждёт 2D-вход: query_emb.reshape(1, -1)
        # index.search -> (scores, positions); собрать результаты по позициям

    def save(self, index_dir: Path) -> None:
        # faiss.write_index + чанки в chunks.json (dataclasses.asdict)

    @classmethod
    def load(cls, index_dir: Path) -> "VectorStore":
        # faiss.read_index + прочитать chunks.json обратно в Chunk
```

`scripts/build_index.py` — линейный скрипт без функций: load_documents → chunk_documents → Embedder → store.add → store.save. В конце напечатать: документов, чанков, размер индекса, время.

**Edge cases:** `k` больше числа векторов (FAISS вернёт позиции `-1` — отфильтруй); поиск по пустому индексу (верни `[]`); `load` из несуществующей папки (понятная ошибка «сначала запусти build_index.py», а не голый traceback).

## Пошаговая инструкция

1. Сначала `add` + `search`, проверь в памяти на 3 игрушечных векторах — до всякого сохранения.
2. Потом `save`/`load` — это скучная сериализация, но именно она делает интерфейс (модуль 07) быстрым: индексация один раз, запуск за секунды.
3. Потом `build_index.py` и настоящий индекс.
4. **Грабли:** рассинхрон эмбеддингов и чанков (отфильтровал одно, забыл другое) — от этого assert в `add`. Позиционное соответствие — единственный «клей» между FAISS и метаданными.
5. **Грабли:** shape запроса. `index.search` хочет `(n_queries, dim)`, а `embed_query` отдаёт `(dim,)` — не забудь reshape.
6. Это момент, когда всплывают ошибки прошлых модулей: если в топе мусор — смотри, что за чанки (слишком длинные? changelog прорвался? — возвращайся в 01/02).

## Как проверить

Мини-assert на игрушечных векторах (в `__main__` vector_store.py):

```python
e = Embedder()
v = e.embed_texts(["cats and dogs", "kittens and puppies", "linear algebra"])
store = VectorStore(); store.add(v, fake_chunks)  # 3 фейковых чанка
top = store.search(e.embed_query("pets"), k=2)
assert {r.chunk.chunk_id for r in top} == {0, 1}  # оба «про животных», не про математику
```

Затем — **смоук-тест retrieval** на реальном индексе (5 вопросов, ответы на которые точно есть в доках):

- "How do I declare optional query parameters?" → жду `tutorial/query-params*`
- "How to return a custom status code?" → `tutorial/response-status-code.md` / `advanced/...`
- "How does dependency injection work?" → `tutorial/dependencies/*`
- "How to handle file uploads?" → `tutorial/request-files.md`
- "How to add CORS?" → `tutorial/cors.md`

Для каждого выведи top-3: `score, doc_path, heading`. Правильный результат: в top-3 есть чанк из ожидаемого файла, scores релевантных ≈ 0.6–0.9.

## Критерий готовности

- [ ] Игрушечный assert проходит; save → load → search даёт те же результаты.
- [ ] `build_index.py` строит индекс с диска до диска за разумное время (минуты).
- [ ] Смоук-тест: минимум 4 из 5 вопросов находят ожидаемый файл в top-3.
- [ ] Могу объяснить, почему IndexFlatIP и что бы я взял на 10 млн векторов.
