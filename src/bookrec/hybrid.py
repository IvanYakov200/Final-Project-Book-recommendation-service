"""Этап 2 (часть 2). Гибридные стратегии ранжирования пула кандидатов.

Все стратегии работают с таблицей кандидатов из candidates.py (строка = пара user/book с
признаками) и возвращают скор для каждой строки. Дальше top-k внутри пользователя.

1. WeightedBlend   : score = Σ_m w_m * z_m   (z_m - стандартизированный скор модели m).
                     Веса подбираются на tune_users максимизацией nDCG@10, глобально или
                     отдельно для каждого сегмента активности.
2. LearnedRanker   : градиентный бустинг / логистическая регрессия на всех признаках
                     (скоры моделей + признаки пользователя, книги и взаимодействия).
                     Обучается на fit_users, метка = релевантность книги в тесте.
3. MMRReranker     : слой разнообразия поверх любого ранжировщика
                     (Maximal Marginal Relevance + штраф за повтор автора).
4. HybridRecommender: итоговый сервис с холодным стартом
                     (новый пользователь -> популярное, онбординг -> Content, иначе -> ранжировщик).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .candidates import SCORE_MODELS
from .metrics import ndcg_at_k, precision_at_k, recall_at_k

SEGMENT_NAMES = {0: "холодные (≤5)", 1: "средние (6-20)", 2: "активные (>20)"}


# ----------------------------------------------------------------------------
# Быстрая оценка на пуле: top-k внутри пользователя + метрики
# ----------------------------------------------------------------------------

def topk_from_pool(pool: pd.DataFrame, scores: np.ndarray, k: int = 10) -> dict:
    """{user_id: [book_id, ...]} - top-k по скору внутри каждого пользователя."""
    df = pd.DataFrame({"u": pool["user_id"].to_numpy(), "b": pool["book_id"].to_numpy(), "s": scores})
    df = df.sort_values(["u", "s"], ascending=[True, False], kind="stable")
    top = df.groupby("u", sort=False).head(k)
    return {u: g["b"].tolist() for u, g in top.groupby("u", sort=False)}


def eval_on_pool(pool: pd.DataFrame, scores: np.ndarray, ds, k: int = 10) -> pd.DataFrame:
    """Per-user Precision/Recall/nDCG@k по top-k из пула (IDCG считается по всем релевантным книгам)."""
    recs = topk_from_pool(pool, scores, k)
    rows = []
    for u, r in recs.items():
        rel = ds.relevant_test[u]
        rows.append({"user_id": u, f"Precision@{k}": precision_at_k(r, rel, k),
                     f"Recall@{k}": recall_at_k(r, rel, k), f"nDCG@{k}": ndcg_at_k(r, rel, k)})
    return pd.DataFrame(rows)


def mean_ndcg(pool, scores, ds, k=10) -> float:
    return float(eval_on_pool(pool, scores, ds, k)[f"nDCG@{k}"].mean())


# ----------------------------------------------------------------------------
# 1. Взвешенное смешивание
# ----------------------------------------------------------------------------

class WeightedBlend:
    """score = Σ w_m z_m. weights: {model: w} или {segment: {model: w}}."""

    def __init__(self, weights: dict, by_segment: bool = False, models=SCORE_MODELS):
        self.weights = weights
        self.by_segment = by_segment
        self.models = tuple(models)

    @staticmethod
    def _blend(pool: pd.DataFrame, w: dict) -> np.ndarray:
        return sum(w.get(m, 0.0) * pool[f"s_{m}"].to_numpy() for m in w)

    def score(self, pool: pd.DataFrame) -> np.ndarray:
        if not self.by_segment:
            return self._blend(pool, self.weights)
        out = np.zeros(len(pool))
        seg = pool["u_segment"].to_numpy().astype(int)
        for s, w in self.weights.items():
            m = seg == s
            if m.any():
                out[m] = self._blend(pool[m], w)
        return out

    # ---------- подбор весов ----------
    @classmethod
    def optimize(cls, pool: pd.DataFrame, ds, models=SCORE_MODELS, n_random: int = 200,
                 n_coord_rounds: int = 2, seed: int = 0, k: int = 10, verbose: bool = True):
        """Случайный поиск по симплексу весов + покоординатное уточнение. Критерий: mean nDCG@k."""
        models = tuple(models)
        rng = np.random.default_rng(seed)
        Z = np.stack([pool[f"s_{m}"].to_numpy() for m in models], axis=1)

        def ndcg_for(w):
            return mean_ndcg(pool, Z @ np.asarray(w), ds, k)

        history = []
        # стартовые точки: каждая модель по отдельности + равные веса + случайные точки симплекса
        starts = [np.eye(len(models))[i] for i in range(len(models))] + [np.ones(len(models)) / len(models)]
        starts += list(rng.dirichlet(np.ones(len(models)), size=n_random))
        best_w, best = None, -1
        for w in starts:
            v = ndcg_for(w)
            history.append((tuple(np.round(w, 3)), v))
            if v > best:
                best, best_w = v, np.asarray(w, float)
        # покоординатное уточнение
        grid = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
        for _ in range(n_coord_rounds):
            for j in range(len(models)):
                for g in grid:
                    w = best_w.copy(); w[j] = g
                    if w.sum() == 0:
                        continue
                    v = ndcg_for(w)
                    if v > best + 1e-6:
                        best, best_w = v, w
        best_w = best_w / best_w.sum()
        if verbose:
            print("  best nDCG@%d = %.4f  weights = %s" % (k, best, dict(zip(models, np.round(best_w, 3)))))
        return dict(zip(models, best_w)), best, history

    @classmethod
    def optimize_by_segment(cls, pool: pd.DataFrame, ds, models=SCORE_MODELS, **kw):
        weights, scores = {}, {}
        for s in sorted(pool["u_segment"].unique()):
            sub = pool[pool["u_segment"] == s]
            if kw.get("verbose", True):
                print(f"[{SEGMENT_NAMES[int(s)]}] users={sub['user_id'].nunique()}")
            w, best, _ = cls.optimize(sub, ds, models, **kw)
            weights[int(s)], scores[int(s)] = w, best
        return weights, scores


def hw_hybrid_scores(pool: pd.DataFrame, beta: float = 20.0) -> np.ndarray:
    """Гибрид из ДЗ №1: w = n/(n+β) для PureSVD, (1-w) для Content. Для сравнения."""
    n = pool["u_n_ratings"].to_numpy()
    w = n / (n + beta)
    return w * pool["s_pure_svd"].to_numpy() + (1 - w) * pool["s_content"].to_numpy()


# ----------------------------------------------------------------------------
# 2. Обучаемый ранжировщик
# ----------------------------------------------------------------------------

class LearnedRanker:
    """Ранжировщик второго уровня: P(релевантна | признаки). kind: 'hgb' или 'logreg'."""

    def __init__(self, feature_cols, kind: str = "hgb", seed: int = 42, **params):
        self.feature_cols = list(feature_cols)
        self.kind = kind
        self.seed = seed
        self.params = params
        self.model = None

    def _make(self):
        if self.kind == "hgb":
            from sklearn.ensemble import HistGradientBoostingClassifier
            p = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=200,
                     l2_regularization=1.0, early_stopping=True, validation_fraction=0.1,
                     n_iter_no_change=30, random_state=self.seed)
            p.update(self.params)
            return HistGradientBoostingClassifier(**p)
        if self.kind == "logreg":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            p = dict(C=1.0, max_iter=2000)
            p.update(self.params)
            return make_pipeline(StandardScaler(), LogisticRegression(**p))
        raise ValueError(self.kind)

    def fit(self, pool: pd.DataFrame, verbose: bool = True):
        t = time.time()
        X = pool[self.feature_cols].to_numpy(dtype=np.float32)
        y = pool["label"].to_numpy()
        self.model = self._make().fit(X, y)
        if verbose:
            n_iter = getattr(self.model, "n_iter_", None)
            print(f"  fitted {self.kind} on {len(pool):,} rows x {len(self.feature_cols)} features"
                  f"{'' if n_iter is None else f', iters={n_iter}'} ({time.time() - t:.0f}s)")
        return self

    def score(self, pool: pd.DataFrame) -> np.ndarray:
        X = pool[self.feature_cols].to_numpy(dtype=np.float32)
        return self.model.predict_proba(X)[:, 1]

    def permutation_importance(self, pool: pd.DataFrame, ds, features=None, k: int = 10, seed: int = 0) -> pd.Series:
        """Падение nDCG@k при перемешивании признака (внутри всего пула). Чем больше, тем важнее."""
        rng = np.random.default_rng(seed)
        features = self.feature_cols if features is None else list(features)
        base = mean_ndcg(pool, self.score(pool), ds, k)
        X = pool[self.feature_cols].to_numpy(dtype=np.float32)
        out = {}
        for f in features:
            j = self.feature_cols.index(f)
            Xp = X.copy(); Xp[:, j] = rng.permutation(Xp[:, j])
            s = self.model.predict_proba(Xp)[:, 1]
            out[f] = base - mean_ndcg(pool, s, ds, k)
        return pd.Series(out, name="drop in nDCG@%d" % k).sort_values(ascending=False)


# ----------------------------------------------------------------------------
# 3. Разнообразие: MMR + штраф за автора
# ----------------------------------------------------------------------------

class MMRReranker:
    """Maximal Marginal Relevance поверх скоров ранжировщика.

    На каждом шаге выбираем книгу, максимизирующую
        λ · rel(i)  −  (1 − λ) · max_{j ∈ selected} sim(i, j)  −  γ · [автор(i) уже в списке],
    где rel - скор ранжировщика, нормированный в [0, 1] внутри списка кандидатов,
    sim - косинус TF-IDF профилей книг (контентная близость).
    λ = 1 возвращает исходное ранжирование; чем меньше λ, тем разнообразнее список.
    """

    def __init__(self, ds, content_model, lam: float = 0.7, author_penalty: float = 0.0, top_n: int = 50):
        self.ds = ds
        self.tfidf = content_model.tfidf
        self.lam = lam
        self.gamma = author_penalty
        self.top_n = top_n
        from .features import book_first_author
        self.first_author = book_first_author(ds)

    def rerank(self, book_ids, scores, k: int = 10) -> list:
        book_ids = np.asarray(book_ids); scores = np.asarray(scores, float)
        order = np.argsort(-scores, kind="stable")[: self.top_n]
        book_ids, scores = book_ids[order], scores[order]
        if self.lam >= 1.0 and self.gamma == 0:
            return book_ids[:k].tolist()
        lo, hi = scores.min(), scores.max()
        rel = (scores - lo) / (hi - lo) if hi > lo else np.ones_like(scores)
        pos = self.ds.item_ids.get_indexer(book_ids)
        S = (self.tfidf[pos] @ self.tfidf[pos].T).toarray()
        authors = self.first_author[pos]

        selected, max_sim, used_authors = [], np.zeros(len(book_ids)), set()
        avail = np.ones(len(book_ids), dtype=bool)
        for _ in range(min(k, len(book_ids))):
            pen = np.array([self.gamma if a in used_authors else 0.0 for a in authors])
            mmr = self.lam * rel - (1 - self.lam) * max_sim - pen
            mmr[~avail] = -np.inf
            i = int(np.argmax(mmr))
            selected.append(i); avail[i] = False; used_authors.add(authors[i])
            max_sim = np.maximum(max_sim, S[i])
        return book_ids[selected].tolist()


# ----------------------------------------------------------------------------
# Адаптеры к Evaluator: recommender(user_id, k) -> [book_id]
# ----------------------------------------------------------------------------

class PoolRecommender:
    """Ранжирует пул кандидатов пользователя; пулы берутся из кэша или строятся на лету."""

    def __init__(self, generator, scorer, pools: dict | None = None, reranker: MMRReranker | None = None):
        self.gen = generator
        self.scorer = scorer            # объект с .score(pool_df) или функция pool_df -> scores
        self.pools = pools or {}        # user_id -> DataFrame кандидатов
        self.reranker = reranker

    def pool_for(self, user_id) -> pd.DataFrame:
        if user_id not in self.pools:
            self.pools[user_id] = self.gen.build(user_id, with_label=False)
        return self.pools[user_id]

    def scores_for(self, user_id):
        pool = self.pool_for(user_id)
        s = self.scorer.score(pool) if hasattr(self.scorer, "score") else self.scorer(pool)
        return pool["book_id"].to_numpy(), np.asarray(s, float)

    def __call__(self, user_id, k: int = 10) -> list:
        book_ids, s = self.scores_for(user_id)
        if self.reranker is not None:
            return self.reranker.rerank(book_ids, s, k)
        top = np.argsort(-s, kind="stable")[:k]
        return book_ids[top].tolist()


def split_pool_by_user(pool: pd.DataFrame) -> dict:
    return {u: g.reset_index(drop=True) for u, g in pool.groupby("user_id", sort=False)}


# ----------------------------------------------------------------------------
# 4. Итоговый сервис с холодным стартом
# ----------------------------------------------------------------------------

class HybridRecommender:
    """
    recommend(user_id=None, liked_books=None, N=10):
      - известный пользователь с историей -> пул кандидатов + ранжировщик (+ MMR);
      - новый пользователь, назвавший любимые книги (онбординг) -> Content + PureSVD по этим книгам;
      - ничего не известно -> байесовская популярность.
    Прочитанные книги исключаются на уровне генерации кандидатов.
    """

    def __init__(self, ds, models, generator, scorer, reranker: MMRReranker | None = None, min_history: int = 1):
        self.ds, self.models, self.gen = ds, models, generator
        self.pool_rec = PoolRecommender(generator, scorer, reranker=reranker)
        self.min_history = min_history

    def recommend(self, user_id=None, liked_books=None, N: int = 10) -> pd.DataFrame:
        if user_id is not None and len(self.ds.seen(user_id)) >= self.min_history:
            ids, source = self.pool_rec(user_id, N), "hybrid ranker"
        elif liked_books:
            pos = self.ds.item_ids.get_indexer(liked_books)
            s = (self.models["content"].score_positions(pos) + 0.5 * self.models["pure_svd"].score_positions(pos)
                 / max(np.abs(self.models["pure_svd"].score_positions(pos)).max(), 1e-9))
            s[pos] = -np.inf
            top = np.argsort(-s)[:N]
            ids, source = self.ds.item_ids[top].tolist(), "onboarding (content + pure_svd)"
        else:
            s = self.models["popularity"].score_user(None)
            ids, source = self.ds.item_ids[np.argsort(-s)[:N]].tolist(), "popular (bayesian)"
        out = self.ds.book_titles.loc[ids].reset_index().rename(columns={"id": "book_id"})
        out.insert(0, "source", source)
        return out
