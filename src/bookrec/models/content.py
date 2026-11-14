"""Контентная модель: TF-IDF по названию и тегам книги (как в ДЗ №1).

Профиль пользователя = средний TF-IDF-вектор книг, которые ему понравились в train.
Скор книги = косинус между TF-IDF книги и профилем пользователя.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BaseModel

# Служебные теги описывают отношение читателя к книге, а не её содержание
SERVICE_TAGS = {
    "to-read", "currently-reading", "favorites", "favourites", "owned", "books-i-own",
    "owned-books", "my-books", "my-library", "i-own", "library", "to-buy", "wish-list",
    "default", "kindle", "ebook", "ebooks", "e-book", "audiobook", "audiobooks", "audio",
    "own", "books", "book", "read", "re-read", "read-in-2016", "read-in-2015", "abandoned",
    "did-not-finish", "dnf", "shelfari-favorites", "favorite-books", "all-time-favorites",
    "to-read-fiction", "my-favorites", "favorite", "on-hold", "maybe", "home-library", "own-it", "to-get",
}


def clean_book_tags(bt: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """Top-N содержательных тегов на книгу: без служебных, без «мусора» и тегов с годом."""
    return (bt[~bt["tag_name"].isin(SERVICE_TAGS)]
            .loc[lambda d: d["tag_name"].str.contains(r"[a-z]")
                 & ~d["tag_name"].str.contains(r"(?:19|20)\d\d")]
            .sort_values(["book_id", "count"], ascending=[True, False])
            .groupby("book_id").head(top_n)
            .assign(token=lambda d: d["tag_name"].str.replace(r"[^a-z0-9]+", "_", regex=True).str.strip("_"))
            .reset_index(drop=True))


class ContentModel(BaseModel):
    name = "content"

    def __init__(self, top_tags_per_book: int = 30):
        super().__init__()
        self.top_tags_per_book = top_tags_per_book

    def fit(self, ds):
        self.attach(ds)
        clean_tags = clean_book_tags(ds.bt, self.top_tags_per_book)
        tag_text = clean_tags.groupby("book_id")["token"].apply(" ".join)
        # books уже в порядке item_ids, поэтому строка i TF-IDF матрицы = книга item_ids[i]
        profile = (ds.books["original_title"].str.lower() + " "
                   + ds.books["id"].map(tag_text).fillna(""))
        self.vectorizer = TfidfVectorizer(token_pattern=r"[a-z0-9_]+", stop_words="english",
                                          min_df=2, max_df=0.5, sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform(profile).tocsr()   # (n_items x vocab), строки нормированы
        self.clean_tags = clean_tags
        return self

    # --- скоры ---
    def profile_vector(self, pos: np.ndarray) -> np.ndarray:
        """Средний TF-IDF вектор по позициям книг."""
        return np.asarray(self.tfidf[pos].mean(axis=0)).ravel()

    def score_positions(self, pos: np.ndarray) -> np.ndarray:
        if len(pos) == 0:
            return np.zeros(self.tfidf.shape[0])
        return np.asarray(self.tfidf @ self.profile_vector(pos)).ravel()

    def score_user(self, user_id) -> np.ndarray:
        return self.score_positions(self.ds.liked(user_id))

    def score_books(self, book_ids) -> np.ndarray:
        """Онбординг: скоры по списку любимых книг нового пользователя."""
        return self.score_positions(self.ds.item_ids.get_indexer(book_ids))

    def similarity_rows(self, pos: np.ndarray) -> np.ndarray:
        """Косинусные близости книг pos со всеми книгами: (len(pos) x n_items)."""
        return (self.tfidf[pos] @ self.tfidf.T).toarray()

    def similar_books(self, book_id, n: int = 5) -> pd.DataFrame:
        pos = self.ds.item_ids.get_loc(book_id)
        sims = self.similarity_rows(np.array([pos]))[0]
        sims[pos] = -1
        top = np.argsort(-sims)[:n]
        res = self.ds.book_titles.loc[self.ds.item_ids[top]].reset_index()
        res["similarity"] = sims[top].round(3)
        return res
