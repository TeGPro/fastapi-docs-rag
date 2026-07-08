# 03 — Эмбеддинги

**Время: ~1.5 часа** · Файл: `rag/embedder.py`

## Цель модуля

На выходе — класс `Embedder`, превращающий тексты в нормализованные векторы. Ты поймёшь, что такое семантическая близость на практике: сможешь численно показать, что «query parameters» ближе к «URL parameters tutorial», чем к «deploying with Docker».

## Теория-минимум

Модель эмбеддингов — маленький трансформер, обученный так, чтобы **cosine similarity** векторов отражала смысловую близость текстов. Это тебе знакомо: та же идея, что вектор `[CLS]` из твоего опыта с rubert-tiny2, только модель дообучена контрастивно специально для сравнения текстов. Берём `sentence-transformers/all-MiniLM-L6-v2`: 384 измерения, ~80 МБ,秒ы инференса на CPU.

Два правила: (1) вопросы и чанки эмбеддятся **одной и той же** моделью, (2) векторы **нормализуем** (длина = 1) — тогда cosine similarity считается простым скалярным произведением, что нужно для FAISS в следующем модуле.

Углубиться: [sbert.net — Semantic Search](https://www.sbert.net/examples/applications/semantic-search/README.html), [карточка модели](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

## Что реализовать

В `rag/config.py`: `EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`, `EMBEDDING_DIM = 384`.

```python
class Embedder:
    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        # загрузить SentenceTransformer один раз; загрузка ~5-10 сек — потому и класс, а не функция

    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        # -> float32, shape (len(texts), 384), L2-нормализованные
        # у encode() есть готовые параметры batch_size, normalize_embeddings, show_progress_bar

    def embed_query(self, query: str) -> np.ndarray:
        # -> shape (384,); тонкая обёртка над embed_texts
```

**Edge cases:** пустой список (`np.empty((0, EMBEDDING_DIM))`, не падение); пустая строка в списке (допустимо — модель вернёт «нулевой смысл», но лучше отфильтровать ещё в chunker'е).

## Пошаговая инструкция

1. Класс маленький — почти всё делает `model.encode()`. Твоя работа: правильные параметры, правильные shape/dtype на выходе и знание, **почему** normalize_embeddings=True (см. теорию).
2. При первом запуске модель скачается с HuggingFace (~80 МБ) — это разово, дальше из кэша.
3. **Грабли:** float64 вместо float32 — FAISS в модуле 04 требует float32, приведи тип сразу здесь (`.astype(np.float32)`), а не «потом где-нибудь».
4. **Грабли:** прогонять чанки по одному в цикле — в 10–30 раз медленнее батча. `encode` сам батчует, просто отдай ему весь список.
5. Замерь время на всех чанках (`time.perf_counter`) — цифра «N чанков за M секунд на CPU» пригодится для README.

## Как проверить

```python
if __name__ == "__main__":
    emb = Embedder()
    v = emb.embed_texts(["How to declare query parameters?",
                         "Query Parameters — tutorial about URL query params",
                         "Deploy FastAPI with Docker containers"])
    assert v.shape == (3, 384) and v.dtype == np.float32
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-3)  # нормализация
    sim_close, sim_far = v[0] @ v[1], v[0] @ v[2]
    print(f"близкая пара: {sim_close:.3f}, далёкая: {sim_far:.3f}")
    assert sim_close > sim_far + 0.1
```

## Критерий готовности

- [ ] Все assert'ы проходят; близкая пара даёт заметно больший cosine, чем далёкая.
- [ ] Векторы float32 и нормализованы.
- [ ] Знаю время эмбеддинга всей коллекции чанков на своём CPU.
- [ ] Могу объяснить, зачем нормализация и почему вопрос и чанк эмбеддятся одной моделью.
