"""Сквозной пайплайн: данные -> предобработка -> признаки -> базовые модели -> кандидаты ->
гибрид (смешивание, бустинг, нейросеть, MMR) -> оценка -> выдача рекомендаций.

Каждый этап кэширует результат в cfg.artifacts_dir, поэтому повторный запуск быстрый, а запуск на
новых данных делается указанием другого data_dir (и, желательно, другого artifacts_dir) или --force.

Запуск из консоли: python run_pipeline.py --data-dir goodsbooks-10k --artifacts-dir artifacts
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .candidates import CandidateGenerator, feature_columns
from .config import Config
from .data import build_dataset
from .features import book_features, user_features
from .hybrid import (HybridRecommender, LearnedRanker, MMRReranker, PoolRecommender, WeightedBlend,
                     hw_hybrid_scores, mean_ndcg, split_pool_by_user)
from .metrics import Evaluator, metric_columns
from .models import fit_all_models


class Pipeline:
    def __init__(self, cfg: Config, log=print):
        self.cfg = cfg
        self.log = log
        self.art = cfg.artifacts_path
        self.timings = {}

    # ------------------------------------------------------------------ util
    def _step(self, name):
        self.log(f"\n=== {name} ===")
        return time.time()

    def _done(self, name, t):
        self.timings[name] = round(time.time() - t, 1)
        self.log(f"    [{name}] {self.timings[name]}s")

    def _cached_pool(self, name, users, force=False):
        p = self.art / f"{name}_pool.parquet"
        if p.exists() and not force:
            return pd.read_parquet(p)
        df = self.gen.build_many(users, verbose=False)
        df.to_parquet(p)
        return df

    # ------------------------------------------------------------------ stages
    def load_data(self):
        t = self._step("1. Загрузка и предобработка данных")
        self.ds = build_dataset(self.cfg)
        self.log("    " + ", ".join(f"{k}={v:,}" for k, v in self.ds.summary().items()))
        self._done("data", t)

    def fit_models(self, force=False):
        t = self._step("2. Базовые модели (Popularity, Content, Item-CF, FunkSVD, PureSVD)")
        self.models = fit_all_models(self.ds, cache_dir=self.art / "models", force=force, verbose=False)
        self._done("models", t)

    def build_features(self):
        t = self._step("3. Признаки пользователей и книг")
        self.book_feats = book_features(self.ds, self.models["content"], self.models["popularity"])
        self.user_feats = user_features(self.ds, self.book_feats)
        self.book_feats.to_parquet(self.art / "book_features.parquet")
        self.user_feats.to_parquet(self.art / "user_features.parquet")
        self.log(f"    book: {self.book_feats.shape}, user: {self.user_feats.shape}")
        self._done("features", t)

    def build_pools(self, force=False):
        t = self._step("4. Пулы кандидатов и признаки взаимодействий (fit / tune / eval)")
        self.gen = CandidateGenerator(self.ds, self.models, self.book_feats, self.user_feats,
                                      pool_per_model=self.cfg.pool_per_model)
        self.train_pool = self._cached_pool("train", self.ds.fit_users, force)
        self.tune_pool = self._cached_pool("tune", self.ds.tune_users, force)
        self.eval_pool = self._cached_pool("eval", self.ds.eval_users, force)
        self.feature_cols = feature_columns(self.train_pool)
        self.log(f"    train {self.train_pool.shape}, tune {self.tune_pool.shape}, eval {self.eval_pool.shape}")
        self._done("pools", t)

    def fit_hybrid(self, force=False, with_nn=True):
        t = self._step("5. Гибрид: веса смешивания, бустинг, нейросеть, MMR")
        ds, art = self.ds, self.art
        # 5a. взвешенное смешивание по сегментам
        p = art / "blend_weights.json"
        if p.exists() and not force:
            w = {int(k): v for k, v in json.load(open(p)).items()}
        else:
            w, _ = WeightedBlend.optimize_by_segment(self.tune_pool, ds, n_random=100, verbose=False)
            json.dump({str(k): {m: float(x) for m, x in v.items()} for k, v in w.items()}, open(p, "w"), indent=1)
        self.blend = WeightedBlend(w, by_segment=True)
        self.log("    blend weights: " + "; ".join(f"seg{s}: " + ", ".join(f"{m}={x:.2f}" for m, x in v.items() if x > 0.05)
                                                   for s, v in w.items()))
        # 5b. бустинг
        p = art / f"ranker_hgb_{len(self.feature_cols)}.joblib"
        if p.exists() and not force:
            self.hgb = joblib.load(p)
        else:
            self.hgb = LearnedRanker(self.feature_cols, kind="hgb").fit(self.train_pool, verbose=False)
            joblib.dump(self.hgb, p)
        self.log(f"    HGB tune nDCG@10 = {mean_ndcg(self.tune_pool, self.hgb.score(self.tune_pool), ds):.4f}")
        # 5c. нейросеть
        self.nn = None
        if with_nn:
            from .neural import NeuralRanker, RankAverage
            p = art / "nn_bce.joblib"
            if p.exists() and not force:
                self.nn = joblib.load(p)
            else:
                c = self.cfg
                self.nn = NeuralRanker(self.feature_cols, hidden=c.nn_hidden, dropout=c.nn_dropout, epochs=c.nn_epochs,
                                       lr=c.nn_lr, batch_size=c.nn_batch_size, loss="bce")
                self.nn.fit(self.train_pool, val_pool=self.tune_pool, ds=ds, verbose=False)
                joblib.dump(self.nn, p)
            self.log(f"    NN  tune nDCG@10 = {mean_ndcg(self.tune_pool, self.nn.score(self.tune_pool), ds):.4f}")
            self.final_scorer = RankAverage({"hgb": self.hgb, "nn": self.nn})
        else:
            self.final_scorer = self.hgb
        # 5d. MMR: максимум разнообразия при потере nDCG <= 1% (на tune_users)
        p = art / "mmr.json"
        if p.exists() and not force:
            mmr = json.load(open(p))
        else:
            pools_tune = split_pool_by_user(self.tune_pool)
            ev = Evaluator(ds, users=ds.tune_users, k_list=self.cfg.k_list)
            for lam, gam in [(1.0, 0.0), (0.9, 0.0), (0.8, 0.0), (0.7, 0.0), (0.9, 0.1), (0.8, 0.1)]:
                rr = MMRReranker(ds, self.models["content"], lam=lam, author_penalty=gam)
                ev.evaluate(f"{lam}|{gam}", PoolRecommender(self.gen, self.final_scorer, pools=pools_tune, reranker=rr), verbose=False)
            tab = ev.summary(content_model=self.models["content"])
            ok = tab[tab["nDCG@10"] >= tab["nDCG@10"].max() * 0.99]
            lam, gam = map(float, ok["ILD"].idxmax().split("|"))
            mmr = {"lam": lam, "author_penalty": gam}
            json.dump(mmr, open(p, "w"))
        self.reranker = MMRReranker(ds, self.models["content"], lam=mmr["lam"], author_penalty=mmr["author_penalty"])
        self.log(f"    MMR: λ={mmr['lam']}, штраф за автора={mmr['author_penalty']}")
        self._done("hybrid", t)

    def evaluate(self, with_base=True):
        t = self._step("6. Оценка на eval_users")
        ds = self.ds
        ev = Evaluator(ds, k_list=self.cfg.k_list)
        if with_base:
            for name, m in self.models.items():
                ev.evaluate(name, m.recommender(), verbose=False)
        pools_eval = split_pool_by_user(self.eval_pool)
        hybrids = {
            "Hybrid: HW (PureSVD+Content)": PoolRecommender(self.gen, hw_hybrid_scores, pools=pools_eval),
            "Hybrid: blend by segment": PoolRecommender(self.gen, self.blend, pools=pools_eval),
            "Hybrid: HGB all features": PoolRecommender(self.gen, self.hgb, pools=pools_eval),
        }
        if self.nn is not None:
            hybrids["Hybrid: NN ranker"] = PoolRecommender(self.gen, self.nn, pools=pools_eval)
            hybrids["Hybrid: HGB + NN"] = PoolRecommender(self.gen, self.final_scorer, pools=pools_eval)
        hybrids["Final: ranker + MMR"] = PoolRecommender(self.gen, self.final_scorer, pools=pools_eval, reranker=self.reranker)
        for name, rec in hybrids.items():
            ev.evaluate(name, rec, verbose=False)
        self.evaluator = ev
        self.summary = ev.summary(content_model=self.models["content"])
        self.by_segment = ev.by_segment("nDCG@10")
        self.summary.to_csv(self.art / "metrics_summary.csv")
        self.by_segment.to_csv(self.art / "metrics_by_segment.csv")
        self.log(self.summary[["Precision@10", "Recall@10", "nDCG@10", "Coverage", "ILD"]].round(4).to_string())
        self._plots()
        self._done("evaluate", t)

    def _plots(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        (self.art / "plots").mkdir(exist_ok=True)
        s = self.summary
        fig, axes = plt.subplots(1, 3, figsize=(16, 0.45 * len(s) + 1.5), sharey=True)
        for ax, col in zip(axes, ["Precision@10", "Recall@10", "nDCG@10"]):
            ax.barh(s.index, s[col].values, color=["#55A868" if i.startswith(("Hybrid", "Final")) else "#4C72B0" for i in s.index])
            for y, v in enumerate(s[col].values):
                ax.text(v, y, f" {v:.3f}", va="center", fontsize=8)
            ax.set_title(col); ax.set_xlim(0, s[col].max() * 1.25)
        axes[0].invert_yaxis(); plt.tight_layout()
        fig.savefig(self.art / "plots" / "metrics.png", dpi=130); plt.close(fig)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        self.by_segment.T.plot(kind="bar", ax=ax, width=0.85, colormap="viridis")
        ax.set_ylabel("nDCG@10"); ax.legend(fontsize=7); plt.xticks(rotation=0); plt.tight_layout()
        fig.savefig(self.art / "plots" / "segments.png", dpi=130); plt.close(fig)

    def build_service(self):
        t = self._step("7. Сервис рекомендаций и примеры выдачи")
        self.service = HybridRecommender(self.ds, self.models, self.gen, self.final_scorer, reranker=self.reranker)
        demo = self.ds.n_train[self.ds.n_train.between(45, 55)].index[0]
        out = pd.concat([
            self.service.recommend(user_id=demo, N=10).assign(case=f"user {demo}"),
            self.service.recommend(liked_books=[1, 126], N=10).assign(case="onboarding: Hunger Games + Dune"),
            self.service.recommend(N=10).assign(case="new user"),
        ])
        out.to_csv(self.art / "sample_recommendations.csv", index=False)
        self.log(out.groupby("case").head(3)[["case", "source", "title"]].to_string(index=False))
        self._done("service", t)

    def write_report(self):
        rep = [f"# Отчёт пайплайна\n", f"data_dir: `{self.cfg.data_dir}`, artifacts: `{self.cfg.artifacts_dir}`\n",
               "## Конфигурация\n```\n" + json.dumps(asdict(self.cfg), ensure_ascii=False, indent=1) + "\n```\n",
               "## Метрики на eval_users\n", self.summary.round(4).to_markdown(), "\n\n## nDCG@10 по сегментам\n",
               self.by_segment.round(4).to_markdown(), "\n\n## Время этапов, с\n",
               pd.Series(self.timings).to_frame("sec").to_markdown(), "\n"]
        (self.art / "report.md").write_text("\n".join(rep), encoding="utf-8")
        self.log(f"\nОтчёт: {self.art / 'report.md'}")

    # ------------------------------------------------------------------ run
    def run(self, force=False, with_nn=True, with_base=True):
        t0 = time.time()
        self.load_data()
        self.fit_models(force)
        self.build_features()
        self.build_pools(force)
        self.fit_hybrid(force, with_nn)
        self.evaluate(with_base)
        self.build_service()
        self.write_report()
        self.log(f"\nГотово за {time.time() - t0:.0f}s")
        return self
