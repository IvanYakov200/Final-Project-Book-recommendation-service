"""Единый интерфейс базовой модели.

Каждая модель умеет:
- fit(ds): обучиться только на ds.train;
- score_user(user_id) -> np.ndarray длины ds.n_items: «чем больше, тем лучше», порядок = ds.item_ids.
  Это ключевое соглашение: раз все модели отдают скоры в одном порядке книг, их можно
  складывать, стандартизировать и подавать в ранжировщик без всякого маппинга ID.
- recommend(user_id, k): top-k непрочитанных book_id.
"""

from __future__ import annotations

import numpy as np


def top_k(scores: np.ndarray, k: int, exclude_pos: np.ndarray | None = None) -> np.ndarray:
    """Позиции k лучших элементов по убыванию scores, исключая exclude_pos."""
    s = scores.astype(float, copy=True)
    if exclude_pos is not None and len(exclude_pos):
        s[exclude_pos] = -np.inf
    k = min(k, len(s))
    top = np.argpartition(-s, k - 1)[:k]
    return top[np.argsort(-s[top], kind="stable")]


class BaseModel:
    name: str = "base"

    def __init__(self):
        self.ds = None

    # --- служебное: ссылку на датасет не сериализуем в кэш ---
    def attach(self, ds):
        self.ds = ds
        return self

    def detach(self):
        self.ds = None

    # --- интерфейс ---
    def fit(self, ds):
        raise NotImplementedError

    def score_user(self, user_id) -> np.ndarray:
        raise NotImplementedError

    def recommend(self, user_id, k: int = 10) -> list:
        pos = top_k(self.score_user(user_id), k, self.ds.seen(user_id))
        return self.ds.item_ids[pos].tolist()

    def recommender(self):
        """Функция (user_id, k) -> [book_id], как ожидает metrics.evaluate."""
        return lambda u, k: self.recommend(u, k)
