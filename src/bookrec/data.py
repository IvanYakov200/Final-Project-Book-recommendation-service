"""Загрузка, очистка и разбиение данных Goodbooks-10k.

Протокол повторяет ДЗ №1, чтобы метрики были сравнимы:
- дубликаты пар (user, book) удаляются, остаётся последняя оценка;
- train/test = 80/20 случайно ВНУТРИ каждого пользователя (времени оценок в данных нет);
- eval_users (3000) и tune_users (2000) выбираются теми же seed, что в ДЗ;
- fit_users (новая выборка) нужны для обучения ранжировщиков второго уровня.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config


# ----------------------------------------------------------------------------
# Загрузка и очистка
# ----------------------------------------------------------------------------

def load_raw(cfg: Config) -> dict[str, pd.DataFrame]:
    d = cfg.data_path
    return {
        "ratings": pd.read_csv(d / "ratings.csv"),
        "books": pd.read_csv(d / "books.csv"),
        "tags": pd.read_csv(d / "tags.csv"),
        "book_tags": pd.read_csv(d / "book_tags.csv"),
    }


def clean_ratings(ratings: pd.DataFrame) -> pd.DataFrame:
    """Убираем дубли (user, book), оставляя последнюю оценку (как в ДЗ)."""
    return (ratings.drop_duplicates(subset=["user_id", "book_id"], keep="last")
                   .reset_index(drop=True))


def clean_books(books: pd.DataFrame) -> pd.DataFrame:
    books = books.copy()
    books["original_title"] = books["original_title"].fillna(books["title"])
    books["authors"] = books["authors"].fillna("")
    return books


def build_tag_table(books: pd.DataFrame, tags: pd.DataFrame, book_tags: pd.DataFrame) -> pd.DataFrame:
    """Теги книг с правильным маппингом ID.

    Ловушка датасета: ratings.book_id == books.id, а book_tags.goodreads_book_id == books.book_id.
    Возвращает таблицу (book_id = books.id, tag_name, count) без артефактов count <= 0.
    """
    bt = (book_tags
          .merge(books[["id", "book_id"]], left_on="goodreads_book_id", right_on="book_id")
          .merge(tags, on="tag_id")
          [["id", "tag_name", "count"]]
          .rename(columns={"id": "book_id"}))
    return bt[bt["count"] > 0].reset_index(drop=True)


# ----------------------------------------------------------------------------
# Разбиение
# ----------------------------------------------------------------------------

def split_per_user(ratings: pd.DataFrame, frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """20% оценок КАЖДОГО пользователя в тест. Пользователи с 1-2 оценками целиком в train."""
    test = ratings.groupby("user_id", group_keys=False).sample(frac=frac, random_state=seed)
    train = ratings.drop(test.index)
    return train.reset_index(drop=True), test.reset_index(drop=True)


# ----------------------------------------------------------------------------
# Контейнер с данными
# ----------------------------------------------------------------------------

@dataclass
class Dataset:
    cfg: Config
    ratings: pd.DataFrame
    books: pd.DataFrame            # отсортированы по id; строка i соответствует item_ids[i]
    bt: pd.DataFrame               # теги: book_id, tag_name, count
    train: pd.DataFrame
    test: pd.DataFrame
    item_ids: pd.Index             # глобальный индекс книг; все модели возвращают скоры в этом порядке

    # производные структуры (заполняются в __post_init__)
    book_titles: pd.DataFrame = field(init=False)
    seen_pos: dict = field(init=False)      # user -> np.array позиций книг из train
    liked_pos: dict = field(init=False)     # user -> np.array позиций понравившихся книг (rating >= порога)
    seen_ids: pd.Series = field(init=False) # user -> set book_id (для метрик)
    n_train: pd.Series = field(init=False)
    relevant_test: pd.Series = field(init=False)   # user -> set релевантных book_id в тесте
    eval_users: np.ndarray = field(init=False)
    tune_users: np.ndarray = field(init=False)
    fit_users: np.ndarray = field(init=False)

    def __post_init__(self):
        cfg = self.cfg
        self.book_titles = self.books.set_index("id")[["title", "authors"]]

        pos = self.item_ids.get_indexer(self.train["book_id"])
        tr = self.train.assign(pos=pos)
        self.seen_pos = {u: g.to_numpy() for u, g in tr.groupby("user_id")["pos"]}
        liked = tr[tr["rating"] >= cfg.relevant_threshold]
        self.liked_pos = {u: g.to_numpy() for u, g in liked.groupby("user_id")["pos"]}
        self.seen_ids = tr.groupby("user_id")["book_id"].apply(set)
        self.n_train = tr.groupby("user_id").size()

        self.relevant_test = (self.test[self.test["rating"] >= cfg.relevant_threshold]
                              .groupby("user_id")["book_id"].apply(set))

        # Выборки пользователей: те же seed, что в ДЗ, чтобы eval/tune совпали с прежними
        users_with_rel = self.relevant_test.index.to_numpy()
        self.eval_users = np.random.default_rng(cfg.seed).choice(users_with_rel, cfg.n_eval_users, replace=False)
        rest = np.setdiff1d(users_with_rel, self.eval_users)
        self.tune_users = np.random.default_rng(7).choice(rest, cfg.n_tune_users, replace=False)
        rest = np.setdiff1d(rest, self.tune_users)
        self.fit_users = np.random.default_rng(11).choice(rest, min(cfg.n_fit_users, len(rest)), replace=False)

    # --- удобные геттеры ---
    @property
    def n_items(self) -> int:
        return len(self.item_ids)

    def seen(self, user_id) -> np.ndarray:
        return self.seen_pos.get(user_id, np.empty(0, dtype=int))

    def liked(self, user_id) -> np.ndarray:
        return self.liked_pos.get(user_id, np.empty(0, dtype=int))

    def summary(self) -> dict:
        return {
            "ratings": len(self.ratings), "users": self.ratings["user_id"].nunique(),
            "books": self.n_items, "train": len(self.train), "test": len(self.test),
            "users_with_relevant_test": len(self.relevant_test),
            "eval_users": len(self.eval_users), "tune_users": len(self.tune_users),
            "fit_users": len(self.fit_users),
        }


def build_dataset(cfg: Config) -> Dataset:
    """Полный шаг «загрузка данных -> предобработка -> split»."""
    raw = load_raw(cfg)
    ratings = clean_ratings(raw["ratings"])
    books = clean_books(raw["books"]).sort_values("id").reset_index(drop=True)
    bt = build_tag_table(books, raw["tags"], raw["book_tags"])
    train, test = split_per_user(ratings, cfg.test_frac, cfg.seed)
    item_ids = pd.Index(books["id"].to_numpy())
    return Dataset(cfg=cfg, ratings=ratings, books=books, bt=bt, train=train, test=test, item_ids=item_ids)
