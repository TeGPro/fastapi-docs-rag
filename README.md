# Ask the Docs — RAG-ассистент по документации FastAPI

QA-система, отвечающая на вопросы по официальной документации FastAPI: семантический поиск по документации + генерация ответа LLM с указанием источников и честным отказом, когда ответа в доках нет.

Пайплайн написан **с нуля, без LangChain/LlamaIndex** — каждый шаг (парсинг Markdown, chunking, эмбеддинги, векторный поиск, промпт, генерация, оценка) реализован руками. Цель — контроль над каждым решением: chunk size, метрика близости, содержимое промпта, метрики качества.

## Что умеет

- **Semantic search** по ~1085 чанкам документации FastAPI (FAISS, косинусная близость).
- **Grounded-ответы**: LLM отвечает строго по найденному контексту, ссылается на источники номерами `[1]`, `[2]`, и **отказывается**, если ответа в доках нет — не выдумывает.
- **CLI и HTTP API**: спросить из консоли или через `POST /ask`.
- **Оценка качества**: hand-labeled набор из 15 вопросов, retrieval hit-rate@5, ручная разметка faithfulness/relevance.

## Результаты оценки

На собственном наборе из 15 вопросов (10 обычных + 3 переформулировки без терминов из заголовков + 2 вопроса-ловушки):

| Метрика | Значение |
|---|---|
| **Retrieval hit-rate@5** | **0.92** (12/13, ловушки в знаменатель не входят) |
| **Ловушки (grounding)** | **2/2** корректный отказ |
| **Faithfulness / Relevance** | размечено вручную по шкале 0/1/2, см. [`data/eval_results.md`](data/eval_results.md) |

Два независимых уровня качества измеряются раздельно: **retrieval** (нашли ли нужную страницу) и **generation** (верен ли ответ найденному и отвечает ли на вопрос).

Ключевое наблюдение — эти оси независимы. Единственный промах retrieval (вопрос «reuse validation logic» — слишком общая формулировка) дал ответ, который **faithful=2** (верен принесённым чанкам), но **relevance=0** (принесена соседняя тема, не dependency injection). Это иллюстрация того, зачем faithfulness и relevance разделяют: ответ может быть честен к контексту и при этом не отвечать на вопрос.

## Архитектура

```
ИНДЕКСАЦИЯ (офлайн, один раз — scripts/build_index.py)

 fastapi/docs/en/docs/**/*.md
        │
        ▼
 [data_loader]  ──►  list[Document]        (парсинг Markdown, очистка мусора)
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

## Ключевые инженерные решения

- **`IndexFlatIP` + нормализованные векторы = косинусная близость.** На ~1085 векторах честный полный перебор — доли миллисекунды; приближённые индексы (IVF/HNSW) оправданы от миллионов векторов.
- **Chunk size 180 слов / overlap 40**, подобрано под лимит эмбеддера: `all-MiniLM-L6-v2` молча обрезает вход после 256 токенов, поэтому потолок чанка — ~200 слов.
- **Heading-aware chunking**: документ режется на секции по заголовкам, длинные секции — скользящим окном; в текст чанка дописывается префикс `title / heading` — дешёвый приём, заметно улучшающий retrieval.
- **Grounded-промптинг**: `temperature=0` для детерминизма; system-промпт запрещает выходить за контекст и требует цитировать источники номерами.
- **Ресурсы грузятся один раз**: модель эмбеддингов и индекс поднимаются при старте (CLI — перед обработкой, API — в lifespan), а не на каждый запрос.

## Стек

- **Python 3.11+**, dataclasses, pathlib
- **sentence-transformers** (`all-MiniLM-L6-v2`, 384-мерные эмбеддинги, CPU)
- **faiss-cpu** — векторный индекс
- **Groq API** (`llama-3.3-70b-versatile`) — генерация ответа, бесплатный тир
- **FastAPI + Pydantic** — HTTP-интерфейс
- Данные: официальная документация FastAPI (Markdown из репозитория `fastapi/fastapi`)

## Структура репозитория

```
mini-rag/
├── README.md
├── requirements.txt
├── rag/
│   ├── config.py          # пути, имена моделей, константы chunking'а
│   ├── data_loader.py     # загрузка и парсинг Markdown
│   ├── chunker.py         # нарезка на чанки
│   ├── embedder.py        # эмбеддинги
│   ├── vector_store.py    # FAISS + метаданные, save/load
│   ├── generator.py       # промпт + вызов Groq
│   ├── evaluate.py        # retrieval hit-rate@k
│   ├── cli.py             # консольный интерфейс
│   └── api.py             # FastAPI-эндпоинт
├── scripts/
│   ├── build_index.py     # офлайн-индексация: docs → FAISS
│   └── smoke_test.py      # ручной прогон retrieval на тестовых вопросах
└── data/
    ├── fastapi/           # git clone fastapi/fastapi (в .gitignore)
    ├── index/             # faiss.index + chunks.json (в .gitignore)
    ├── eval_questions.json # набор вопросов для оценки
    └── eval_results.md    # результаты оценки с ручной разметкой
```

## Как запустить

```bash
# 1. Зависимости
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Данные — документация FastAPI
git clone --depth 1 https://github.com/fastapi/fastapi data/fastapi

# 3. Ключ Groq (бесплатно: https://console.groq.com)
export GROQ_API_KEY="gsk_..."

# 4. Построить индекс
python scripts/build_index.py
#    -> 147 документов, 1085 чанков, индекс на диске, ~12 сек

# 5. Спросить из консоли
python -m rag.cli "How do I declare optional query parameters?" --k 5

# или поднять API
uvicorn rag.api:app
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How do I return a custom status code?"}'
# -> {"answer": "...", "sources": ["tutorial/response-status-code.md", ...]}

# 6. Оценка качества
python -m rag.evaluate
```

Swagger UI доступен на `localhost:8000/docs`.

## Пример

```
$ python -m rag.cli "How do I handle file uploads?"

To handle file uploads, define a file parameter with type UploadFile [2], which
gives access to metadata like filename and content type [2]. To receive uploaded
files you need to install python-multipart [5].

Sources:
  tutorial/request-files.md (score=0.514)
  reference/uploadfile.md (score=0.495)
```

На вопрос вне документации («What is the capital of France?», «integrate with Django ORM») система отвечает отказом, а не выдумкой — grounding работает.