"""Этап 2 (часть 1). Генерация кандидатов и таблица признаков (user, book).

Для каждого пользователя:
1. каждая базовая модель отдаёт скоры для всех книг (один порядок = ds.item_ids);
2. уже прочитанные книги исключаются;
3. берём top-N от КАЖДОЙ модели и объединяем в общий пул кандидатов;
4. для каждого кандидата собираем признаки:
   - s_<model>     : z-скор модели (стандартизирован по всем непрочитанным книгам пользователя);
   - rank_<model>  : log(1 + ранг) книги в выдаче модели;
   - n_votes       : сколько моделей включили книгу в свой top-N;
   - x_*           : взаимодействие с историей пользователя (близость в трёх пространствах,
                     совпадение автора, совпадение жанрового профиля, «мейнстримность» и год
                     относительно привычек пользователя);
   - u_*, b_*      : признаки пользователя и книги (features.py);
   - label         : 1, если книга релевантна в тесте пользователя (для обучения/оценки).

Пул кандидатов нужен, потому что ранжировщик второго уровня дорого применять ко всем 10 000
книгам для каждого пользователя, а релевантные книги почти всегда попадают в объединённый top-N
(см. анализ recall пула в ноутбуке).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .features import book_first_author
from .models.base import top_k

SCORE_MODELS = ("popularity", "content", "itemcf", "funk_svd", "pure_svd")


def zscore_unseen(x: np.ndarray, seen: np.ndarray) -> np.ndarray:
    """Стандартизация скора по непрочитанным книгам (у моделей разные шкалы)."""
    mask = np.ones(len(x), dtype=bool)
    mask[seen] = False
    v = x[mask]
    std = v.std()
    return (x - v.mean()) / (std if std > 1e-12 else 1.0)


class CandidateGenerator:
    def __init__(self, ds, models: dict, book_feats: pd.DataFrame, user_feats: pd.DataFrame,
                 pool_per_model: int = 50, generators=SCORE_MODELS):
        self.ds = ds
        self.models = models
        self.book_feats = book_feats
        self.user_feats = user_feats
        self.pool_per_model = pool_per_model
        self.generators = tuple(generators)

        self.bf = book_feats.to_numpy(dtype=np.float32)
        self.bcols = list(book_feats.columns)
        self.first_author = book_first_author(ds)
        gcols = [c for c in self.bcols if c.startswith("b_genre_")]
        self.G = self.bf[:, [self.bcols.index(c) for c in gcols]]
        self.G_norm = self.G / np.maximum(np.linalg.norm(self.G, axis=1, keepdims=True), 1e-9)
        self.ucols = list(user_feats.columns)
        self.u_genre_cols = [c for c in self.ucols if c.startswith("u_genre_")]
        self.default_user = user_feats.median().to_numpy(dtype=np.float32)

    # ------------------------------------------------------------------
    # Скоры всех моделей для пользователя
    # ------------------------------------------------------------------
    def user_scores(self, user_id) -> dict:
        """Сырые скоры каждой модели + вспомогательные близости к истории."""
        m = self.models
        out = {}
        for name in SCORE_MODELS:
            if name == "itemcf":
                cf = m["itemcf"].score_all(user_id)
                out["itemcf"] = cf["evidence"]
                out["_itemcf_max_sim"] = cf["max_sim"]
            else:
                out[name] = m[name].score_user(user_id)
        liked = self.ds.liked(user_id)
        content = m["content"]
        out["_content_max_sim"] = (content.similarity_rows(liked).max(axis=0) if len(liked)
                                   else np.zeros(self.ds.n_items))
        out["_puresvd_max_sim"] = m["pure_svd"].max_sim_to_liked(user_id)
        return out

    # ------------------------------------------------------------------
    # Пул кандидатов
    # ------------------------------------------------------------------
    def pool(self, user_id, scores: dict | None = None):
        """Позиции кандидатов и матрица «кто предложил» (n_cand x n_generators)."""
        scores = scores or self.user_scores(user_id)
        seen = self.ds.seen(user_id)
        tops = {g: top_k(scores[g], self.pool_per_model, seen) for g in self.generators}
        cand = np.unique(np.concatenate(list(tops.values())))
        votes = np.stack([np.isin(cand, tops[g]) for g in self.generators], axis=1)
        return cand, votes, scores

    # ------------------------------------------------------------------
    # Таблица признаков
    # ------------------------------------------------------------------
    def build(self, user_id, with_label: bool = True) -> pd.DataFrame:
        cand, votes, scores = self.pool(user_id)
        seen, liked = self.ds.seen(user_id), self.ds.liked(user_id)
        n = len(cand)
        cols = {}

        # --- скоры и ранги моделей ---
        for name in SCORE_MODELS:
            z = zscore_unseen(scores[name], seen)
            cols[f"s_{name}"] = z[cand]
            order = np.argsort(-np.where(np.isin(np.arange(self.ds.n_items), seen), -np.inf, scores[name]), kind="stable")
            rank = np.empty(self.ds.n_items, dtype=np.int32); rank[order] = np.arange(self.ds.n_items)
            cols[f"rank_{name}"] = np.log1p(rank[cand]).astype(np.float32)
        cols["n_votes"] = votes.sum(axis=1).astype(np.float32)

        # --- взаимодействия с историей пользователя ---
        cols["x_content_max_sim"] = scores["_content_max_sim"][cand]
        cols["x_itemcf_max_sim"] = scores["_itemcf_max_sim"][cand]
        cols["x_puresvd_max_sim"] = scores["_puresvd_max_sim"][cand]
        liked_authors = set(self.first_author[liked]) if len(liked) else set()
        cols["x_same_author"] = np.array([a in liked_authors for a in self.first_author[cand]], dtype=np.float32)
        seen_authors = set(self.first_author[seen]) if len(seen) else set()
        cols["x_author_read"] = np.array([a in seen_authors for a in self.first_author[cand]], dtype=np.float32)

        if user_id in self.user_feats.index:
            uvec = self.user_feats.loc[user_id].to_numpy(dtype=np.float32)
        else:
            uvec = self.default_user
        u = dict(zip(self.ucols, uvec))
        ug = np.array([u[c] for c in self.u_genre_cols], dtype=np.float32)
        ug_norm = ug / max(np.linalg.norm(ug), 1e-9)
        cols["x_genre_match"] = self.G_norm[cand] @ ug_norm
        bi = {c: self.bcols.index(c) for c in ("b_log_gr_ratings", "b_year", "b_is_series", "b_mean_rating")}
        cols["x_pop_vs_user"] = self.bf[cand, bi["b_log_gr_ratings"]] - u["u_mean_book_pop"]
        cols["x_year_vs_user"] = (self.bf[cand, bi["b_year"]] - u["u_mean_book_year"]) / 10.0
        cols["x_rating_vs_user"] = self.bf[cand, bi["b_mean_rating"]] - u["u_mean_rating"]

        df = pd.DataFrame(cols)
        # --- признаки пользователя и книги ---
        udf = pd.DataFrame(np.repeat(uvec[None, :], n, axis=0), columns=self.ucols)
        bdf = pd.DataFrame(self.bf[cand], columns=self.bcols)
        df = pd.concat([df, udf, bdf], axis=1)

        df.insert(0, "book_id", self.ds.item_ids[cand].to_numpy())
        df.insert(0, "user_id", user_id)
        if with_label:
            rel = self.ds.relevant_test.get(user_id, set())
            df["label"] = df["book_id"].isin(rel).astype(np.int8)
        return df

    def build_many(self, users, with_label: bool = True, verbose: bool = True) -> pd.DataFrame:
        t = time.time()
        parts = []
        for j, u in enumerate(users):
            parts.append(self.build(u, with_label))
            if verbose and (j + 1) % 1000 == 0:
                print(f"    {j + 1}/{len(users)} users, {time.time() - t:.0f}s", flush=True)
        df = pd.concat(parts, ignore_index=True)
        return df

    # ------------------------------------------------------------------
    # Диагностика пула
    # ------------------------------------------------------------------
    def pool_recall(self, users) -> pd.DataFrame:
        """Какую долю релевантных книг покрывает пул (и каждая модель по отдельности)."""
        rows = []
        for u in users:
            cand, votes, _ = self.pool(u)
            rel = self.ds.relevant_test.get(u, set())
            if not rel:
                continue
            rel_pos = self.ds.item_ids.get_indexer(list(rel))
            in_pool = np.isin(rel_pos, cand)
            row = {"pool_size": len(cand), "recall_pool": in_pool.mean()}
            for j, g in enumerate(self.generators):
                row[f"recall_{g}"] = np.isin(rel_pos, cand[votes[:, j]]).mean()
            rows.append(row)
        return pd.DataFrame(rows)


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ("user_id", "book_id", "label")]
