"""Быстрая проверка признаков и генератора кандидатов на небольшой выборке."""
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

pd.set_option("display.width", 200, "display.max_columns", 40)
cfg = Config()
ds = build_dataset(cfg)
models = fit_all_models(ds, cache_dir=cfg.artifacts_path / "models", verbose=False)

t = time.time()
bf = book_features(ds, models["content"], models["popularity"])
print(f"book_features: {bf.shape} ({time.time()-t:.1f}s)")
print(bf.describe().T[["mean", "std", "min", "max"]].round(3))
gcols = [c for c in bf.columns if c.startswith("b_genre_")]
print("\nДоля книг с жанром > 0.05:")
print((bf[gcols] > 0.05).mean().round(3).sort_values(ascending=False))
print("\nПример: id=1 (Hunger Games)")
print(bf.loc[1, gcols][bf.loc[1, gcols] > 0.02].round(3))

t = time.time()
uf = user_features(ds, bf)
print(f"\nuser_features: {uf.shape} ({time.time()-t:.1f}s)")
print(uf.describe().T[["mean", "std", "min", "max"]].round(3))

gen = CandidateGenerator(ds, models, bf, uf, pool_per_model=cfg.pool_per_model)
t = time.time()
df = gen.build_many(ds.eval_users[:200], verbose=False)
print(f"\ncandidates for 200 users: {df.shape} ({time.time()-t:.1f}s, {(time.time()-t)/200*1000:.0f} ms/user)")
print("pool size per user:", df.groupby("user_id").size().describe().round(1).to_dict())
print("label rate:", df["label"].mean().round(4), " users with >=1 positive:", (df.groupby("user_id")["label"].sum() > 0).mean())
print("groups:", {k: len(v) for k, v in feature_groups(feature_columns(df)).items()})
print(df[[c for c in df.columns if c.startswith(("s_", "rank_", "x_")) or c == "n_votes"]].describe().T[["mean", "std", "min", "max"]].round(3))

t = time.time()
pr = gen.pool_recall(ds.eval_users[:300])
print(f"\npool recall (300 users, {time.time()-t:.1f}s):")
print(pr.mean().round(3))
print("NaNs:", df.isna().sum().sum())
