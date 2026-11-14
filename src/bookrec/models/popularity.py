"""Неперсонализированная модель: байесовский (взвешенный, формула IMDb) рейтинг.

WR = v/(v+m) * R + m/(v+m) * C,  R - средний рейтинг книги, v - число оценок,
C - среднее по всем книгам, m - 90-й перцентиль числа оценок.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import BaseModel


class PopularityModel(BaseModel):
    name = "popularity"

    def __init__(self, min_ratings: int = 50, m_quantile: float = 0.9):
        super().__init__()
        self.min_ratings = min_ratings
        self.m_quantile = m_quantile

    def fit(self, ds):
        self.attach(ds)
        stats = ds.train.groupby("book_id")["rating"].agg(R="mean", v="count")
        C = ds.train["rating"].mean()
        m = stats["v"].quantile(self.m_quantile)
        stats["weighted_rating"] = stats["v"] / (stats["v"] + m) * stats["R"] + m / (stats["v"] + m) * C
        stats["mean_min"] = np.where(stats["v"] >= self.min_ratings, stats["R"], np.nan)
        stats = stats.reindex(ds.item_ids)
        self.stats = stats
        # Скоры одинаковы для всех пользователей. Книг без оценок в train нет, но на всякий случай -> C
        self.scores_ = stats["weighted_rating"].fillna(C).to_numpy()
        self.table_ = stats
        return self

    def score_user(self, user_id) -> np.ndarray:
        return self.scores_

    def top(self, n: int = 10) -> pd.DataFrame:
        return (self.table_.sort_values("weighted_rating", ascending=False).head(n)
                .join(self.ds.book_titles))
