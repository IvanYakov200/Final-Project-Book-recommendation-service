"""Проверка нейросетевого ранжировщика на кэшированных пулах."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import joblib
from bookrec import Config
from bookrec.data import build_dataset
from bookrec.candidates import feature_columns
from bookrec.hybrid import mean_ndcg, LearnedRanker
from bookrec.neural import NeuralRanker, RankAverage

cfg = Config()
ds = build_dataset(cfg)
train_pool = pd.read_parquet(cfg.artifacts_path / "train_pool.parquet")
tune_pool = pd.read_parquet(cfg.artifacts_path / "tune_pool.parquet")
eval_pool = pd.read_parquet(cfg.artifacts_path / "eval_pool.parquet")
cols = feature_columns(train_pool)

res = {}
for loss in ["bce", "pairwise"]:
    print(f"== {loss} ==")
    nn_r = NeuralRanker(cols, hidden=cfg.nn_hidden, dropout=cfg.nn_dropout, epochs=12, lr=cfg.nn_lr,
                        batch_size=cfg.nn_batch_size, loss=loss)
    nn_r.fit(train_pool, val_pool=tune_pool, ds=ds)
    res[loss] = nn_r
    print(f"  eval nDCG@10 = {mean_ndcg(eval_pool, nn_r.score(eval_pool), ds):.4f}")
    joblib.dump(nn_r, cfg.artifacts_path / f"nn_{loss}.joblib")

hgb_path = cfg.artifacts_path / f"ranker_hgb_{len(cols)}.joblib"
if hgb_path.exists():
    hgb = joblib.load(hgb_path)
    print("hgb eval nDCG@10 =", round(mean_ndcg(eval_pool, hgb.score(eval_pool), ds), 4))
    for loss, nn_r in res.items():
        ens = RankAverage({"hgb": hgb, "nn": nn_r})
        print(f"ensemble hgb + nn_{loss}: tune {mean_ndcg(tune_pool, ens.score(tune_pool), ds):.4f}  eval {mean_ndcg(eval_pool, ens.score(eval_pool), ds):.4f}")
