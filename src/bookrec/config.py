"""Единая конфигурация пайплайна. Все «магические числа» живут здесь."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # --- данные ---
    data_dir: str = "goodsbooks-10k"
    artifacts_dir: str = "artifacts"
    seed: int = 42
    test_frac: float = 0.2            # доля оценок каждого пользователя, уходящая в тест
    relevant_threshold: int = 4       # оценка >= 4 считается «понравилось»

    # --- выборки пользователей для оценки (протокол из ДЗ №1) ---
    n_eval_users: int = 3000          # итоговые метрики
    n_tune_users: int = 2000          # подбор весов гибрида / гиперпараметров
    n_fit_users: int = 6000           # обучение ранжировщиков второго уровня
    k_list: tuple = (5, 10)

    # --- базовые модели ---
    top_tags_per_book: int = 30
    itemcf_k: int = 20
    itemcf_shrink: float = 50.0
    pure_svd_factors: int = 500
    funk_svd_params: dict = field(default_factory=lambda: dict(
        n_factors=10, reg_all=0.1, n_epochs=30, random_state=42))
    popularity_min_ratings: int = 50

    # --- генерация кандидатов и гибрид ---
    pool_per_model: int = 50          # сколько кандидатов берём от каждой модели
    n_genres_user_profile: int = 16

    # --- нейросетевой ранжировщик ---
    nn_hidden: tuple = (128, 64)
    nn_epochs: int = 15
    nn_lr: float = 1e-3
    nn_batch_size: int = 2048
    nn_dropout: float = 0.2

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def artifacts_path(self) -> Path:
        p = Path(self.artifacts_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
