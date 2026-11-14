"""Матричные разложения из ДЗ №1.

FunkSVDModel  - явные оценки, surprise.SVD:  r̂ = μ + b_u + b_i + q_iᵀ p_u. Лучший RMSE,
                но для Top-N на уровне Random (всем выдаёт книги с большим b_i).
PureSVDModel  - неявная матрица «понравилось / нет», усечённое SVD (Cremonesi et al., 2010):
                score_u = m_u V Vᵀ. Лучшая одиночная модель Top-N в ДЗ.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from .base import BaseModel


class FunkSVDModel(BaseModel):
    name = "funk_svd"

    def __init__(self, n_factors=10, reg_all=0.1, n_epochs=30, random_state=42):
        super().__init__()
        self.params = dict(n_factors=n_factors, reg_all=reg_all, n_epochs=n_epochs, random_state=random_state)

    def fit(self, ds):
        from surprise import SVD, Dataset as SDataset, Reader

        self.attach(ds)
        reader = Reader(rating_scale=(1, 5))
        data = SDataset.load_from_df(ds.train[["user_id", "book_id", "rating"]], reader)
        trainset = data.build_full_trainset()
        model = SVD(**self.params).fit(trainset)

        # Переносим параметры в numpy, выровненные по глобальному индексу книг
        self.mu = trainset.global_mean
        self.user_index = pd.Index([trainset.to_raw_uid(u) for u in range(trainset.n_users)])
        self.bu = model.bu.copy()
        self.pu = model.pu.copy()
        raw_items = np.array([trainset.to_raw_iid(i) for i in range(trainset.n_items)])
        pos = ds.item_ids.get_indexer(raw_items)
        self.bi = np.zeros(ds.n_items); self.bi[pos] = model.bi
        self.qi = np.zeros((ds.n_items, self.params["n_factors"])); self.qi[pos] = model.qi
        return self

    def score_user(self, user_id) -> np.ndarray:
        if user_id not in self.user_index:
            return self.mu + self.bi
        u = self.user_index.get_loc(user_id)
        return self.mu + self.bu[u] + self.bi + self.qi @ self.pu[u]

    def predict_many(self, df: pd.DataFrame) -> np.ndarray:
        """Предсказания оценок (для RMSE), с обрезкой в [1, 5]."""
        pos = self.ds.item_ids.get_indexer(df["book_id"])
        known = df["user_id"].isin(self.user_index).to_numpy()
        pred = np.full(len(df), self.mu) + self.bi[pos]
        uidx = self.user_index.get_indexer(df["user_id"][known])
        pred[known] += self.bu[uidx] + np.einsum("ij,ij->i", self.qi[pos[known]], self.pu[uidx])
        return np.clip(pred, 1, 5)


class PureSVDModel(BaseModel):
    name = "pure_svd"

    def __init__(self, n_factors: int = 500, seed: int = 42):
        super().__init__()
        self.n_factors = n_factors
        self.seed = seed

    def fit(self, ds):
        self.attach(ds)
        liked = ds.train[ds.train["rating"] >= ds.cfg.relevant_threshold]
        self.user_ids = pd.Index(ds.train["user_id"].unique())
        u = self.user_ids.get_indexer(liked["user_id"])
        i = ds.item_ids.get_indexer(liked["book_id"])
        self.M = sp.csr_matrix((np.ones(len(liked)), (u, i)), shape=(len(self.user_ids), ds.n_items))
        v0 = np.random.default_rng(self.seed).random(min(self.M.shape))
        _, sigma, Vt = svds(self.M, k=self.n_factors, v0=v0)
        self.sigma = sigma
        self.V = Vt.T.astype(np.float32)                                   # (n_items x d)
        self.V_norm = self.V / np.maximum(np.linalg.norm(self.V, axis=1, keepdims=True), 1e-12)
        return self

    def score_positions(self, pos: np.ndarray) -> np.ndarray:
        """Скоры по произвольному набору понравившихся книг (онбординг)."""
        if len(pos) == 0:
            return np.zeros(self.ds.n_items)
        m = np.zeros(self.ds.n_items, dtype=np.float32); m[pos] = 1.0
        return ((m @ self.V) @ self.V.T).astype(float)

    def score_user(self, user_id) -> np.ndarray:
        if user_id not in self.user_ids:
            return np.zeros(self.ds.n_items)
        m_u = self.M[self.user_ids.get_loc(user_id)]
        return np.asarray((m_u @ self.V) @ self.V.T).ravel().astype(float)

    def max_sim_to_liked(self, user_id) -> np.ndarray:
        """Макс. косинус между книгой и понравившимися книгами пользователя в пространстве факторов."""
        pos = self.ds.liked(user_id)
        if len(pos) == 0:
            return np.zeros(self.ds.n_items)
        return (self.V_norm[pos] @ self.V_norm.T).max(axis=0).astype(float)

    def similar_books(self, book_id, n: int = 5) -> pd.DataFrame:
        pos = self.ds.item_ids.get_loc(book_id)
        sims = self.V_norm @ self.V_norm[pos]
        sims[pos] = -np.inf
        top = np.argsort(-sims)[:n]
        res = self.ds.book_titles.loc[self.ds.item_ids[top]].reset_index()
        res["similarity"] = sims[top].round(3)
        return res
