# Ask the Docs — RAG-ассистент по документации FastAPI

QA-система, отвечающая на вопросы по официальной документации FastAPI: семантический поиск по документации + генерация ответа LLM с указанием источников. Пайплайн написан **с нуля, без LangChain/LlamaIndex** — каждый шаг (парсинг, chunking, эмбеддинги, векторный поиск, промпт, генерация, оценка) реализован руками.

> Учебный проект на 2 дня. Инструкции по каждому модулю — в [docs/](docs/), план работы — в [PLAN.md](PLAN.md).

## Архитектура

```
ИНДЕКСАЦИЯ (офлайн, один раз — scripts/build_index.py)

 fastapi/docs/en/docs/**/*.md
        │
        ▼
 [data_loader]  ──►  list[Document]        (парсинг Markdown, очистка)
        │
        ▼
 [chunker]      ──►  list[Chunk]           (нарезка по заголовкам + скользящее окно)
        │
        ▼
 [embedder]     ──►  np.ndarray (n, 384)   (sentence-transformers, all-MiniLM-L6-v2)
        │
        ▼
 [vector_store] ──►  data/index/           (FAISS IndexFlatIP + chunks.json)


ОТВЕТ НА ВОПРОС (онлайн — rag/cli.py или rag/api.py)

 question ──► [embedder] ──► query vector
                                  │
                                  ▼
              [vector_store.search] ──► top-k chunks
                                  │
                                  ▼
              [generator] ──► промпт с контекстом ──► Groq LLM ──► answer + sources
```

## Стек

- **Python 3.11+**, dataclasses, pathlib
- **sentence-transformers** (`all-MiniLM-L6-v2`, 384-мерные эмбеддинги, работает на CPU)
- **faiss-cpu** — векторный индекс
- **Groq API** (`llama-3.3-70b-versatile`) — генерация ответа, бесплатный тир
- **FastAPI + Pydantic** — HTTP-интерфейс
- Данные: официальная документация FastAPI (Markdown из репозитория `fastapi/fastapi`)

## Структура репозитория

```
ask-the-docs/
├── README.md
├── PLAN.md                # план на 2 дня, чекпоинты, scope-cut
├── requirements.txt
├── docs/                  # учебные инструкции по модулям (00–07)
├── rag/
│   ├── __init__.py
│   ├── config.py          # пути, имена моделей, константы chunking'а
│   ├── data_loader.py     # модуль 01: загрузка и парсинг Markdown
│   ├── chunker.py         # модуль 02: нарезка на чанки
│   ├── embedder.py        # модуль 03: эмбеддинги
│   ├── vector_store.py    # модуль 04: FAISS + метаданные
│   ├── generator.py       # модуль 05: промпт + Groq
│   ├── evaluate.py        # модуль 06: оценка качества
│   ├── cli.py             # модуль 07: CLI
│   └── api.py             # модуль 07: FastAPI-эндпоинт
├── scripts/
│   └── build_index.py     # офлайн-индексация: docs → FAISS
└── data/
    ├── fastapi_repo/      # git clone fastapi/fastapi (в .gitignore)
    ├── index/             # faiss.index + chunks.json (в .gitignore)
    └── eval_questions.json
```

## Как запустить

```bash
# 1. Зависимости
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Данные — документация FastAPI
git clone --depth 1 https://github.com/fastapi/fastapi data/fastapi_repo

# 3. Ключ Groq (бесплатно: https://console.groq.com)
export GROQ_API_KEY="gsk_..."

# 4. Построить индекс (~2–3 мин на CPU)
python scripts/build_index.py

# 5. Спросить
python -m rag.cli "How do I declare optional query parameters?"

# или поднять API
uvicorn rag.api:app --reload
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How do I return a custom status code?"}'

# 6. Оценка качества
python -m rag.evaluate
```

## Как описать проект в резюме

Буллеты в стиле достижений (для англоязычного резюме):

- Built a **Retrieval-Augmented Generation (RAG) QA system from scratch** (no LangChain) over the full FastAPI documentation: Markdown parsing, heading-aware chunking, semantic search, and LLM answer generation with source citation.
- Implemented **semantic retrieval** with sentence-transformers embeddings and a FAISS vector index; tuned chunk size/overlap against embedding-model context limits.
- Designed **grounded prompting**: the LLM (Llama 3.3 70B via Groq API) answers strictly from retrieved context, cites sources, and explicitly refuses when the answer is not in the docs.
- Created an **evaluation harness**: a hand-labeled question set, retrieval hit-rate@k, and manual faithfulness/relevance scoring of generated answers.
- Shipped both a **CLI and a FastAPI endpoint** (Pydantic-validated request/response, index loaded once at startup).

Русская версия — переформулируй те же пункты; ключевые слова: RAG с нуля, семантический поиск, FAISS, chunking trade-offs, grounded-промптинг, оценка faithfulness/relevance.
