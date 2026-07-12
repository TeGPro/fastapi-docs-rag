# Результаты оценки RAG

**hit-rate@5 = 12/13 = 0.92** (ловушки в знаменатель не входят)
**Ловушки: 2/2 корректный отказ.**

Шкала: **faithful** — всё ли в ответе подтверждается найденными чанками (2=да, 1=в основном, 0=выдумал).
**relevance** — отвечает ли на заданный вопрос (2=прямо, 1=частично, 0=не про то).

| #  | question                     | hit@5 | faithful | relevance | комментарий |
|----|------------------------------|-------|----------|-----------|-------------|
| 1  | query parameter optional     | ✅    | 2        | 2         |             |
| 2  | read value from URL path     | ✅    | 2        | 2         |             |
| 3  | receive uploaded file        | ✅    | 2        | 2         |             |
| 4  | configure CORS               | ✅    | 2        | 2         |             |
| 5  | custom error 404             | ✅    | 2        | 2         |             |
| 6  | dependency injection         | ✅    | 2        | 2         |             |
| 7  | custom status code           | ✅    | 2        | 2         |             |
| 8  | add middleware               | ✅    | 2        | 2         |             |
| 9  | websocket connection         | ✅    | 2        | 2         |             |
| 10 | deploy with Docker           | ✅    | 2        | 2         |             |
| 11 | browser blocks cross-domain  | ✅    | 2        | 2         | переформулировка cors без CORS — попала |
| 12 | user sends image via form    | ✅    | 2        | 2         | переформулировка request-files — попала |
| 13 | reuse validation logic       | ❌    | 2        | 0         | faithful (в чанках правда) но не про DI → relevance=0 |

## Ловушки (grounding)

| #  | question                          | отказ? |
|----|-----------------------------------|--------|
| 14 | integrate with Django ORM         | ✅     |
| 15 | schedule cron jobs inside FastAPI | ✅     |

## Выводы

- hit-rate@5 = 0.92 (12 из 13). Retrieval находит нужную страницу почти всегда.
- Единственный промах — слишком общий вопрос «переиспользовать логику»: система поняла его шире и принесла соседнюю тему. Чем конкретнее вопрос, тем точнее поиск.
- Обе вопроса-ловушки получили честный отказ — система не выдумывает, когда ответа в доках нет.