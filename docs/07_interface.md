# 07 — Интерфейс: CLI и FastAPI

**Время: ~1.5 часа** · Файлы: `rag/cli.py`, `rag/api.py`

## Цель модуля

На выходе — два тонких входа в уже готовый пайплайн: консольная команда и HTTP-эндпоинт `/ask`. Нового про RAG здесь почти нет — модуль про инженерную гигиену: тяжёлые объекты (модель эмбеддингов, индекс) загружаются **один раз** при старте, а не на каждый запрос.

## Теория-минимум

Вся логика уже написана в модулях 03–05; интерфейс лишь связывает `Embedder` + `VectorStore.load` + `answer_question`. Для CLI хватает stdlib `argparse`. Для FastAPI тебе всё знакомо; единственное новое — инициализация ресурсов при старте приложения через lifespan (или, проще, модульные глобальные переменные — для учебного проекта допустимо).

Углубиться: [FastAPI — Lifespan Events](https://fastapi.tiangolo.com/advanced/events/) — заодно проверишь свой RAG вопросом «how do I run startup logic in FastAPI?».

## Что реализовать

`rag/cli.py`:

```python
def main() -> None:
    # argparse: позиционный question, опция --k (default TOP_K)
    # загрузить Embedder + VectorStore.load(INDEX_DIR)
    # answer_question -> напечатать ответ, затем блок "Sources:" —
    #   уникальные doc_path с score
```

Запуск: `python -m rag.cli "How do I handle file uploads?" --k 5` (нужен блок `if __name__ == "__main__": main()`).

`rag/api.py`:

```python
class AskRequest(BaseModel):
    question: str          # min_length=3
    k: int = TOP_K         # ge=1, le=20

class AskResponse(BaseModel):
    answer: str
    sources: list[str]     # уникальные doc_path

# lifespan: embedder, store загружаются при старте
# POST /ask: AskRequest -> AskResponse
```

**Edge cases:** индекс не построен — понятное сообщение «run scripts/build_index.py first» и на CLI, и при старте API (а не traceback на первом запросе); пустой/слишком короткий вопрос — отдай Pydantic-валидации (`min_length`), не пиши ручных проверок; дубликаты в sources (несколько чанков одного файла) — схлопнуть, сохранив порядок.

## Пошаговая инструкция

1. Сначала CLI — 20 строк, сразу даёт «продукт», которым можно пользоваться.
2. Потом API. Проверь руками, что модель НЕ загружается на каждый запрос: второй запрос должен отвечать заметно быстрее первого холодного старта процесса (загрузка один раз), время ответа ≈ время Groq-вызова.
3. **Грабли:** создать `Embedder()` внутри обработчика запроса — +10 секунд к каждому ответу. Это главная (и единственная серьёзная) ошибка модуля.
4. **Грабли:** `python rag/cli.py` сломается на относительных импортах — запускай как модуль: `python -m rag.cli`.
5. Открой `localhost:8000/docs` — Swagger UI. Скриншот отсюда хорошо смотрится в README.

## Как проверить

```bash
python -m rag.cli "How to add CORS to a FastAPI app?"
# -> связный ответ + Sources: tutorial/cors.md (score=...)

uvicorn rag.api:app
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "How do I return a custom status code?"}'
# -> {"answer": "...", "sources": ["tutorial/response-status-code.md", ...]}

curl -X POST localhost:8000/ask -H "Content-Type: application/json" -d '{"question": ""}'
# -> 422 (валидация Pydantic)
```

## Критерий готовности

- [ ] CLI отвечает на произвольный вопрос и печатает источники.
- [ ] `/ask` работает; пустой вопрос → 422; `/docs` открывается.
- [ ] Модели загружаются один раз при старте (второй запрос быстрый).
- [ ] Ответ API содержит источники — фронт мог бы показать «откуда это».
