"""Базовые модели из ДЗ №1, приведённые к единому интерфейсу (см. base.py)."""

from __future__ import annotations

from pathlib import Path

import joblib

from .base import BaseModel, top_k
from .content import ContentModel
from .itemcf import ItemCFModel
from .popularity import PopularityModel
from .svd import FunkSVDModel, PureSVDModel

__all__ = ["BaseModel", "top_k", "ContentModel", "ItemCFModel", "PopularityModel",
           "FunkSVDModel", "PureSVDModel", "fit_or_load", "fit_all_models"]


def fit_or_load(model: BaseModel, ds, cache_dir: Path | None = None, force: bool = False) -> BaseModel:
    """Обучить модель или взять готовую из кэша artifacts/models/<name>.joblib."""
    if cache_dir is not None:
        path = Path(cache_dir) / f"{model.name}.joblib"
        if path.exists() and not force:
            m = joblib.load(path)
            m.attach(ds)
            return m
    model.fit(ds)
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        model.detach()
        joblib.dump(model, Path(cache_dir) / f"{model.name}.joblib")
        model.attach(ds)
    return model


def fit_all_models(ds, cache_dir: Path | None = None, force: bool = False, verbose: bool = True) -> dict[str, BaseModel]:
    """Все базовые модели ДЗ, обученные только на ds.train."""
    cfg = ds.cfg
    specs = [
        PopularityModel(min_ratings=cfg.popularity_min_ratings),
        ContentModel(top_tags_per_book=cfg.top_tags_per_book),
        ItemCFModel(k=cfg.itemcf_k, shrink=cfg.itemcf_shrink),
        FunkSVDModel(**cfg.funk_svd_params),
        PureSVDModel(n_factors=cfg.pure_svd_factors, seed=cfg.seed),
    ]
    models = {}
    for m in specs:
        if verbose:
            print(f"  [{m.name}] ...", end=" ", flush=True)
        models[m.name] = fit_or_load(m, ds, cache_dir, force)
        if verbose:
            print("ok")
    return models
