"""Метрики ранжирования и протокол оценки (как в ДЗ №1).

Precision@K = hits / K,  Recall@K = hits / |rel|,
nDCG@K = DCG@K / IDCG@K,  DCG@K = Σ rel_i / log2(i + 1).
Плюс «здоровье» выдачи: Coverage (доля каталога в чьих-то рекомендациях),
Personalization (1 - среднее пересечение выдач двух случайных пользователей),
Intra-list diversity (1 - средняя попарная контентная близость внутри списка).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------
# Метрики для одного пользователя
# ----------------------------------------------------------------------------

def precision_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / k


def recall_at_k(recs, relevant, k):
    return len(set(recs[:k]) & relevant) / len(relevant)


def ndcg_at_k(recs, relevant, k):
    dcg = sum(1 / np.log2(i + 2) for i, b in enumerate(recs[:k]) if b in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg


def metric_columns(k_list):
    return [f"{m}@{k}" for k in k_list for m in ("Precision", "Recall", "nDCG")]


# ----------------------------------------------------------------------------
# Оценка рекомендателя на выборке пользователей
# ----------------------------------------------------------------------------

class Evaluator:
    """Считает метрики для функций recommender(user_id, k) -> [book_id] на одних и тех же пользователях."""

    def __init__(self, ds, users=None, k_list=(5, 10)):
        self.ds = ds
        self.users = ds.eval_users if users is None else np.asarray(users)
        self.k_list = tuple(k_list)
        self.k_max = max(k_list)
        self.recs: dict[str, dict] = {}        # model -> {user -> recs}
        self.per_user = pd.DataFrame()

    def evaluate(self, name: str, recommender, verbose: bool = True) -> pd.DataFrame:
        rel_all = self.ds.relevant_test
        rows, recs = [], {}
        for u in self.users:
            r = list(recommender(u, self.k_max))
            recs[u] = r
            rel = rel_all[u]
            row = {"model": name, "user_id": u}
            for k in self.k_list:
                row[f"Precision@{k}"] = precision_at_k(r, rel, k)
                row[f"Recall@{k}"] = recall_at_k(r, rel, k)
                row[f"nDCG@{k}"] = ndcg_at_k(r, rel, k)
            rows.append(row)
        df = pd.DataFrame(rows)
        df["n_train"] = df["user_id"].map(self.ds.n_train).fillna(0).astype(int)
        df["segment"] = segment_users(df["n_train"])
        self.recs[name] = recs
        if len(self.per_user):
            self.per_user = self.per_user[self.per_user["model"] != name]
        self.per_user = pd.concat([self.per_user, df], ignore_index=True)
        if verbose:
            m = df[metric_columns(self.k_list)].mean()
            print(f"{name:32s} P@10={m['Precision@10']:.4f}  R@10={m['Recall@10']:.4f}  nDCG@10={m['nDCG@10']:.4f}")
        return df

    def evaluate_many(self, recommenders: dict, verbose: bool = True):
        for name, rec in recommenders.items():
            self.evaluate(name, rec, verbose)
        return self.summary()

    # --- сводки ---
    def beyond_accuracy(self, name: str, content_model=None) -> pd.Series:
        lists = [self.recs[name][u] for u in self.users]
        unique = {b for r in lists for b in r}
        pairs = np.random.default_rng(0).integers(0, len(lists), size=(2000, 2))
        overlap = np.mean([len(set(lists[a]) & set(lists[b])) / self.k_max for a, b in pairs])
        out = {"Coverage": len(unique) / self.ds.n_items, "Personalization": 1 - overlap}
        if content_model is not None:
            out["ILD"] = intra_list_diversity(lists[:1000], self.ds, content_model)
        return pd.Series(out)

    def summary(self, content_model=None) -> pd.DataFrame:
        cols = metric_columns(self.k_list)
        models = list(dict.fromkeys(self.per_user["model"]))
        s = self.per_user.groupby("model", sort=False)[cols].mean()
        extra = pd.DataFrame({m: self.beyond_accuracy(m, content_model) for m in models}).T
        return s.join(extra)

    def by_segment(self, metric="nDCG@10", models=None) -> pd.DataFrame:
        seg = self.per_user.pivot_table(index="model", columns="segment", values=metric,
                                        aggfunc="mean", observed=False)
        seg = seg.loc[list(dict.fromkeys(self.per_user["model"]))]
        return seg if models is None else seg.loc[models]

    def ci_table(self, metric="nDCG@10", n_boot=1000) -> pd.DataFrame:
        ci = {name: bootstrap_ci(g[metric].to_numpy(), n_boot)
              for name, g in self.per_user.groupby("model", sort=False)}
        return pd.DataFrame(ci, index=[f"{metric} low", f"{metric} high"]).T

    def paired_diff(self, model_a: str, model_b: str, metric="nDCG@10") -> pd.Series:
        a = self.per_user[self.per_user["model"] == model_a].set_index("user_id")[metric]
        b = self.per_user[self.per_user["model"] == model_b].set_index("user_id")[metric]
        d = (a - b.loc[a.index]).to_numpy()
        low, high = bootstrap_ci(d)
        return pd.Series({"mean diff": d.mean(), "CI low": low, "CI high": high,
                          "A better": (d > 0).mean(), "A worse": (d < 0).mean()})


def segment_users(n_train: pd.Series) -> pd.Series:
    return pd.cut(n_train, bins=[-1, 5, 20, 10**6], labels=["холодные (≤5)", "средние (6-20)", "активные (>20)"])


def bootstrap_ci(values, n_boot=1000, seed=0):
    r = np.random.default_rng(seed)
    values = np.asarray(values)
    means = [r.choice(values, len(values)).mean() for _ in range(n_boot)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def intra_list_diversity(rec_lists, ds, content_model) -> float:
    """1 - средняя попарная косинусная близость (TF-IDF) книг внутри списка, усреднённая по спискам."""
    vals = []
    for recs in rec_lists:
        if len(recs) < 2:
            continue
        pos = ds.item_ids.get_indexer(recs)
        S = (content_model.tfidf[pos] @ content_model.tfidf[pos].T).toarray()
        n = len(pos)
        vals.append(1 - (S.sum() - np.trace(S)) / (n * (n - 1)))
    return float(np.mean(vals)) if vals else np.nan


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
