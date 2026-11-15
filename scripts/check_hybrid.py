"""Проверка гибридных стратегий: строим пулы для tune/eval, подбираем веса, обучаем ранжировщик."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from bookrec import Config
from bookrec.data import build_dataset
from bookrec.models import fit_all_models
from bookrec.features import book_features, user_features, feature_groups
from bookrec.candidates import CandidateGenerator, feature_columns
from bookrec.hybrid import (WeightedBlend, LearnedRanker, MMRReranker, PoolRecommender, hw_hybrid_scores,
                            mean_ndcg, eval_on_pool, split_pool_by_user)
from bookrec.metrics import Evaluator

pd.set_option("display.width", 200, "display.max_columns", 40)
cfg = Config()
ds = build_dataset(cfg)
models = fit_all_models(ds, cache_dir=cfg.artifacts_path / "models", verbose=False)
bf = book_features(ds, models["content"], models["popularity"])
uf = user_features(ds, bf)
gen = CandidateGenerator(ds, models, bf, uf, pool_per_model=cfg.pool_per_model)


def cached_pool(name, users):
    p = cfg.artifacts_path / f"{name}_pool.parquet"
    if p.exists():
        return pd.read_parquet(p)
    t = time.time()
    df = gen.build_many(users, verbose=False)
    df.to_parquet(p)
    print(f"built {name}: {df.shape} ({time.time()-t:.0f}s)")
    return df


train_pool = cached_pool("train", ds.fit_users)
tune_pool = cached_pool("tune", ds.tune_users)
eval_pool = cached_pool("eval", ds.eval_users)
print("pools:", train_pool.shape, tune_pool.shape, eval_pool.shape)

# --- одиночные модели и гибрид ДЗ на пуле (tune) ---
print("\n== tune: single models on pool ==")
for m in ["content", "itemcf", "pure_svd"]:
    print(f"  {m:10s} nDCG@10 = {mean_ndcg(tune_pool, tune_pool[f's_{m}'].to_numpy(), ds):.4f}")
print(f"  HW hybrid  nDCG@10 = {mean_ndcg(tune_pool, hw_hybrid_scores(tune_pool), ds):.4f}")

# --- взвешенное смешивание ---
print("\n== weighted blend (global) ==")
t = time.time()
w_glob, best_glob, hist = WeightedBlend.optimize(tune_pool, ds, n_random=150)
print(f"  ({time.time()-t:.0f}s)")
print("\n== weighted blend (by segment) ==")
t = time.time()
w_seg, best_seg = WeightedBlend.optimize_by_segment(tune_pool, ds, n_random=100)
print(f"  ({time.time()-t:.0f}s)")
blend_g = WeightedBlend(w_glob)
blend_s = WeightedBlend(w_seg, by_segment=True)
print(f"  eval: blend global {mean_ndcg(eval_pool, blend_g.score(eval_pool), ds):.4f}, "
      f"by segment {mean_ndcg(eval_pool, blend_s.score(eval_pool), ds):.4f}, "
      f"HW hybrid {mean_ndcg(eval_pool, hw_hybrid_scores(eval_pool), ds):.4f}")

# --- обучаемый ранжировщик + ablation по группам признаков ---
groups = feature_groups(feature_columns(train_pool))
sets = {
    "scores only": groups["scores"],
    "scores + interaction": groups["scores"] + groups["interaction"],
    "all features": feature_columns(train_pool),
}
print("\n== learned ranker (HGB) ==")
rankers = {}
for name, cols in sets.items():
    r = LearnedRanker(cols, kind="hgb").fit(train_pool)
    rankers[name] = r
    print(f"  {name:22s} tune nDCG@10 = {mean_ndcg(tune_pool, r.score(tune_pool), ds):.4f}   "
          f"eval nDCG@10 = {mean_ndcg(eval_pool, r.score(eval_pool), ds):.4f}")
lr = LearnedRanker(feature_columns(train_pool), kind="logreg").fit(train_pool)
print(f"  {'logreg all':22s} tune nDCG@10 = {mean_ndcg(tune_pool, lr.score(tune_pool), ds):.4f}   "
      f"eval nDCG@10 = {mean_ndcg(eval_pool, lr.score(eval_pool), ds):.4f}")

# --- MMR ---
best = rankers["all features"]
pools_eval = split_pool_by_user(eval_pool)
ev = Evaluator(ds)
print("\n== MMR on eval (ranker = all features) ==")
for lam in [1.0, 0.9, 0.8, 0.7, 0.5]:
    rr = MMRReranker(ds, models["content"], lam=lam, author_penalty=0.0)
    rec = PoolRecommender(gen, best, pools=pools_eval, reranker=rr)
    ev.evaluate(f"hgb + MMR λ={lam}", rec, verbose=False)
print(ev.summary(content_model=models["content"])[["Precision@10", "Recall@10", "nDCG@10", "Coverage", "ILD"]].round(4))
