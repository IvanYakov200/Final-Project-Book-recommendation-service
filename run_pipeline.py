#!/usr/bin/env python3
"""Сквозной пайплайн рекомендательной системы: от CSV до рекомендаций и отчёта.

Примеры:
  python3 run_pipeline.py                                  # полный прогон на goodsbooks-10k
  python3 run_pipeline.py --quick --artifacts-dir artifacts_quick   # быстрый smoke-test (~3-5 мин)
  python3 run_pipeline.py --data-dir new_data --artifacts-dir artifacts_new   # новые данные того же формата
  python3 run_pipeline.py --force                          # пересчитать всё, игнорируя кэш

Формат входных данных: папка с ratings.csv, books.csv, tags.csv, book_tags.csv (как в Goodbooks-10k).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from bookrec import Config
from bookrec.pipeline import Pipeline


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="goodsbooks-10k")
    ap.add_argument("--artifacts-dir", default="artifacts")
    ap.add_argument("--force", action="store_true", help="пересчитать модели и пулы, игнорируя кэш")
    ap.add_argument("--no-nn", action="store_true", help="без нейросетевого ранжировщика")
    ap.add_argument("--no-base", action="store_true", help="не оценивать базовые модели (быстрее)")
    ap.add_argument("--quick", action="store_true", help="уменьшенные выборки пользователей для быстрой проверки")
    ap.add_argument("--n-fit-users", type=int)
    ap.add_argument("--n-eval-users", type=int)
    ap.add_argument("--n-tune-users", type=int)
    ap.add_argument("--pool-per-model", type=int)
    ap.add_argument("--nn-epochs", type=int)
    a = ap.parse_args()

    cfg = Config(data_dir=a.data_dir, artifacts_dir=a.artifacts_dir)
    if a.quick:
        cfg.n_fit_users, cfg.n_tune_users, cfg.n_eval_users, cfg.nn_epochs = 1500, 500, 800, 8
    for k in ["n_fit_users", "n_eval_users", "n_tune_users", "pool_per_model", "nn_epochs"]:
        if getattr(a, k) is not None:
            setattr(cfg, k, getattr(a, k))

    Pipeline(cfg).run(force=a.force, with_nn=not a.no_nn, with_base=not a.no_base)


if __name__ == "__main__":
    main()
