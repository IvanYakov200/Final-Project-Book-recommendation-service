"""Item-Based Collaborative Filtering (из ДЗ №1).

Схожесть книг: adjusted cosine (оценки центрированы по пользователю) с shrinkage n/(n+λ).
Для Top-N ранжируем по «силе доказательств» Σ sim(i,j) * (r_uj - r̄_u), а не по предсказанной
оценке: в ДЗ это дало nDCG@10 0.14 против 0.001.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .base import BaseModel


class ItemCFModel(BaseModel):
    name = "itemcf"

    def __init__(self, k: int = 20, shrink: float = 50.0):
        super().__init__()
        self.k = k
        self.shrink = shrink

    def fit(self, ds):
        self.attach(ds)
        df = ds.train
        self.user_ids = pd.Index(df["user_id"].unique())
        u = self.user_ids.get_indexer(df["user_id"])
        i = ds.item_ids.get_indexer(df["book_id"])          # сразу глобальные позиции книг
        r = df["rating"].to_numpy(dtype=float)
        shape = (len(self.user_ids), ds.n_items)

        self.R = sp.csr_matrix((r, (u, i)), shape=shape)
        cnt = np.diff(self.R.indptr)
        self.user_mean = np.asarray(self.R.sum(axis=1)).ravel() / np.maximum(cnt, 1)

        X = sp.csc_matrix((r - self.user_mean[u], (u, i)), shape=shape)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=0)).ravel())
        norms[norms == 0] = 1.0
        self.X = (X @ sp.diags(1.0 / norms)).tocsc()
        self.B = sp.csc_matrix((np.ones_like(r), (u, i)), shape=shape)
        return self

    def similarity_rows(self, pos: np.ndarray) -> np.ndarray:
        """Строки матрицы схожестей для книг pos: (len(pos) x n_items)."""
        S = (self.X[:, pos].T @ self.X).toarray()
        if self.shrink:
            n_common = (self.B[:, pos].T @ self.B).toarray()
            S *= n_common / (n_common + self.shrink)
        return S

    def _user_items(self, uidx: int):
        s, e = self.R.indptr[uidx], self.R.indptr[uidx + 1]
        return self.R.indices[s:e], self.R.data[s:e]

    def score_all(self, user_id) -> dict:
        """prediction, evidence и max_sim (макс. схожесть с прочитанным) для всех книг."""
        n = self.ds.n_items
        if user_id not in self.user_ids:
            z = np.zeros(n)
            return {"prediction": z, "evidence": z, "max_sim": z}
        uidx = self.user_ids.get_loc(user_id)
        items, rates = self._user_items(uidx)
        S = self.similarity_rows(items)
        S[S < 0] = 0
        max_sim = S.max(axis=0) if len(items) else np.zeros(n)
        if len(items) > self.k:
            kth = -np.partition(-S, self.k - 1, axis=0)[self.k - 1]
            S[S < kth] = 0
        sim_sum = S.sum(axis=0)
        pred = np.where(sim_sum > 0, (rates @ S) / np.maximum(sim_sum, 1e-12), self.user_mean[uidx])
        evidence = (rates - self.user_mean[uidx]) @ S
        return {"prediction": pred, "evidence": evidence, "max_sim": max_sim}

    def score_user(self, user_id) -> np.ndarray:
        return self.score_all(user_id)["evidence"]

    def predict_many(self, df: pd.DataFrame) -> pd.Series:
        """Предсказанные оценки для пар (user_id, book_id) — для RMSE."""
        out = pd.Series(np.nan, index=df.index)
        known = df["user_id"].isin(self.user_ids)
        for uid, g in df[known].groupby("user_id"):
            pred = self.score_all(uid)["prediction"]
            out[g.index] = pred[self.ds.item_ids.get_indexer(g["book_id"])]
        return out

    def similar_books(self, book_id, n: int = 5) -> pd.DataFrame:
        pos = self.ds.item_ids.get_loc(book_id)
        sims = self.similarity_rows(np.array([pos]))[0]
        sims[pos] = -np.inf
        top = np.argsort(-sims)[:n]
        res = self.ds.book_titles.loc[self.ds.item_ids[top]].reset_index()
        res["similarity"] = sims[top].round(3)
        return res
