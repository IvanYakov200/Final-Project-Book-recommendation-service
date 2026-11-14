"""Проверка: модели из пакета воспроизводят метрики ДЗ №1 на тех же eval_users."""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from bookrec import Config
from bookrec.data import build_dataset
from bookrec.models import fit_all_models
from bookrec.metrics import Evaluator

cfg = Config()
t = time.time()
ds = build_dataset(cfg)
print("dataset:", ds.summary(), f"({time.time()-t:.1f}s)")
print("eval_users[:5] =", ds.eval_users[:5], " tune_users[:5] =", ds.tune_users[:5])

t = time.time()
models = fit_all_models(ds, cache_dir=cfg.artifacts_path / "models")
print(f"models fitted ({time.time()-t:.1f}s)")

ev = Evaluator(ds)
t = time.time()
summary = ev.evaluate_many({name: m.recommender() for name, m in models.items()})
print(f"eval ({time.time()-t:.1f}s)")

ref = pd.Series({"popularity": 0.002, "content": 0.129, "itemcf": 0.137, "funk_svd": 0.001, "pure_svd": 0.163},
                name="nDCG@10 (ДЗ)")
print(summary.round(4).join(ref))
print(ev.by_segment().round(4))
