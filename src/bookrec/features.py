"""Этап 1. Расширенные признаки пользователей, книг и взаимодействий.

Все признаки считаются ТОЛЬКО по train (и по метаданным книг), иначе была бы утечка теста.

Группы:
- book_features(ds, content)   : популярность, разброс оценок, год, серия, жанровые категории из тегов;
- user_features(ds, book_feats): активность, средняя оценка и её разброс, «мейнстримность» вкуса,
                                 жанровый профиль и его энтропия;
- взаимодействия (user, book)  : считаются в candidates.py, т.к. требуют скоры моделей:
                                 близость к истории пользователя в TF-IDF / Item-CF / PureSVD
                                 пространствах, совпадение автора, совпадение жанрового профиля.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

# Жанровые категории: регулярные выражения по токенам тегов (после очистки в ContentModel)
GENRE_PATTERNS: dict[str, str] = {
    "fiction":        r"^fiction$|general_fiction|literary_fiction|^contemporary$|literature",
    "fantasy":        r"fantasy|magic|dragons|paranormal|urban_fantasy",
    "science_fiction": r"sci_?fi|science_fiction|dystopia|dystopian|space|cyberpunk",
    "romance":        r"romance|romantic|chick_lit|^love$",
    "young_adult":    r"young_adult|^ya$|^ya_|_ya$|teen",
    "mystery_thriller": r"mystery|thriller|crime|suspense|detective",
    "classics":       r"classic",
    "nonfiction":     r"non_?fiction",
    "historical":     r"historical",
    "childrens":      r"children|kids|picture_book|juvenile|middle_grade",
    "horror":         r"horror|vampires?|zombies?|ghost|supernatural",
    "biography_memoir": r"biography|memoir|autobiography",
    "history_politics": r"^history$|_history$|^history_|politics|war$|_war_|wwii",
    "science_psych_business": r"^science$|psychology|business|self_help|philosophy|economics|^math",
    "poetry_drama":   r"poetry|poems?|^plays?$|drama|theatre|shakespeare",
    "comics_graphic": r"comics?|graphic_novels?|manga|superheroes",
    "religion":       r"religion|christian|spirituality|bible|faith|theology",
    "humor":          r"humor|humour|comedy|funny",
}
GENRES = list(GENRE_PATTERNS)


# ----------------------------------------------------------------------------
# Жанры книг из тегов
# ----------------------------------------------------------------------------

def book_genre_matrix(clean_tags: pd.DataFrame, item_ids: pd.Index) -> pd.DataFrame:
    """Мягкий жанровый профиль книги: доля «голосов» (count) её топ-тегов, попавших в жанр.

    Возвращает DataFrame (n_items x n_genres) в порядке item_ids, значения в [0, 1].
    Один тег может попасть в несколько жанров (например, 'ya_fantasy' -> young_adult и fantasy).
    """
    tokens = clean_tags["token"]
    out = pd.DataFrame(0.0, index=item_ids, columns=GENRES)
    total = clean_tags.groupby("book_id")["count"].sum()
    for g, pat in GENRE_PATTERNS.items():
        mask = tokens.str.contains(pat, regex=True)
        s = clean_tags[mask].groupby("book_id")["count"].sum()
        out[g] = (s / total.loc[s.index]).reindex(item_ids).fillna(0.0)
    return out.clip(upper=1.0)


# ----------------------------------------------------------------------------
# Признаки книг
# ----------------------------------------------------------------------------

_SERIES_RE = re.compile(r"\(.*#\s*\d")


def book_features(ds, content_model, popularity_model=None) -> pd.DataFrame:
    """Признаки книг, строка i = книга ds.item_ids[i]."""
    books = ds.books.set_index("id").loc[ds.item_ids]
    tr = ds.train.groupby("book_id")["rating"]
    thr = ds.cfg.relevant_threshold

    f = pd.DataFrame(index=ds.item_ids)
    # Популярность и качество по train
    f["b_n_ratings"] = tr.size().reindex(ds.item_ids).fillna(0)
    f["b_mean_rating"] = tr.mean().reindex(ds.item_ids)
    f["b_std_rating"] = tr.std().reindex(ds.item_ids)
    f["b_share_high"] = tr.apply(lambda s: (s >= thr).mean()).reindex(ds.item_ids)
    if popularity_model is not None:
        f["b_bayes_rating"] = popularity_model.scores_
    # Популярность на Goodreads (длинный хвост, которого нет в обрезанном ratings.csv)
    f["b_log_gr_ratings"] = np.log1p(books["ratings_count"].to_numpy())
    f["b_gr_avg_rating"] = books["average_rating"].to_numpy()
    f["b_gr_std_proxy"] = _rating_spread(books)
    # Метаданные
    year = books["original_publication_year"]
    f["b_year"] = year.fillna(year.median()).clip(lower=1800).to_numpy()
    f["b_is_series"] = books["title"].fillna("").map(lambda t: bool(_SERIES_RE.search(t))).astype(float).to_numpy()
    first_author = books["authors"].str.split(",").str[0].str.strip()
    f["b_author_n_books"] = first_author.map(first_author.value_counts()).to_numpy()
    # Жанры
    genres = book_genre_matrix(content_model.clean_tags, ds.item_ids)
    genres.columns = [f"b_genre_{g}" for g in genres.columns]
    f = pd.concat([f, genres], axis=1)

    fill = {"b_mean_rating": ds.train["rating"].mean(), "b_std_rating": 0.0, "b_share_high": 0.0}
    return f.fillna(fill).astype(np.float32)


def book_first_author(ds) -> np.ndarray:
    """Первый автор каждой книги в порядке ds.item_ids (для признака «совпадение автора»)."""
    books = ds.books.set_index("id").loc[ds.item_ids]
    return books["authors"].str.split(",").str[0].str.strip().to_numpy()


def _rating_spread(books: pd.DataFrame) -> np.ndarray:
    """Разброс оценок на Goodreads по колонкам ratings_1..ratings_5 (std распределения)."""
    cols = [f"ratings_{i}" for i in range(1, 6)]
    if not all(c in books.columns for c in cols):
        return np.zeros(len(books))
    counts = books[cols].to_numpy(dtype=float)
    p = counts / np.maximum(counts.sum(axis=1, keepdims=True), 1)
    vals = np.arange(1, 6)
    mean = p @ vals
    return np.sqrt(p @ vals**2 - mean**2)


# ----------------------------------------------------------------------------
# Признаки пользователей
# ----------------------------------------------------------------------------

def user_features(ds, book_feats: pd.DataFrame) -> pd.DataFrame:
    """Признаки пользователей по train. Индекс = user_id."""
    thr = ds.cfg.relevant_threshold
    tr = ds.train.copy()
    pos = ds.item_ids.get_indexer(tr["book_id"])
    bf = book_feats.to_numpy()
    col = {c: i for i, c in enumerate(book_feats.columns)}

    tr["high"] = (tr["rating"] >= thr).astype(float)
    tr["book_pop"] = bf[pos, col["b_log_gr_ratings"]]
    tr["book_year"] = bf[pos, col["b_year"]]
    tr["book_series"] = bf[pos, col["b_is_series"]]
    tr["book_mean"] = bf[pos, col["b_mean_rating"]]
    g = tr.groupby("user_id")

    f = pd.DataFrame(index=g.size().index)
    f["u_n_ratings"] = g.size()
    f["u_log_n_ratings"] = np.log1p(f["u_n_ratings"])
    f["u_mean_rating"] = g["rating"].mean()
    f["u_std_rating"] = g["rating"].std().fillna(0)
    f["u_share_high"] = g["high"].mean()
    f["u_n_liked"] = g["high"].sum()
    # Насколько пользователь «строже» среднего по книгам, которые он читал (аналог b_u)
    f["u_strictness"] = (g["rating"].mean() - g["book_mean"].mean())
    f["u_mean_book_pop"] = g["book_pop"].mean()      # мейнстримность вкуса
    f["u_mean_book_year"] = g["book_year"].mean()
    f["u_series_share"] = g["book_series"].mean()
    seg = pd.cut(f["u_n_ratings"], bins=[-1, 5, 20, 10**6], labels=[0, 1, 2]).astype(int)
    f["u_segment"] = seg

    # Жанровый профиль: среднее жанровых векторов понравившихся книг (если их нет, всех прочитанных)
    gcols = [c for c in book_feats.columns if c.startswith("b_genre_")]
    G = bf[:, [col[c] for c in gcols]]
    liked = tr[tr["high"] == 1]
    prof_liked = pd.DataFrame(G[ds.item_ids.get_indexer(liked["book_id"])], index=liked["user_id"]).groupby(level=0).mean()
    prof_all = pd.DataFrame(G[pos], index=tr["user_id"]).groupby(level=0).mean()
    prof = prof_liked.reindex(f.index)
    prof = prof.fillna(prof_all.reindex(f.index))
    prof.columns = [c.replace("b_genre_", "u_genre_") for c in gcols]
    p = prof.to_numpy()
    p_norm = p / np.maximum(p.sum(axis=1, keepdims=True), 1e-9)
    f["u_taste_entropy"] = -(p_norm * np.log(p_norm + 1e-12)).sum(axis=1)   # разнообразие вкуса
    f = pd.concat([f, prof], axis=1)
    return f.astype(np.float32)


# ----------------------------------------------------------------------------
# Утилиты
# ----------------------------------------------------------------------------

def feature_groups(columns) -> dict[str, list[str]]:
    """Разбиение признаков на группы для ablation-анализа."""
    cols = list(columns)
    return {
        "scores": [c for c in cols if c.startswith("s_") or c.startswith("rank_") or c in ("n_votes",)],
        "user": [c for c in cols if c.startswith("u_")],
        "book": [c for c in cols if c.startswith("b_")],
        "interaction": [c for c in cols if c.startswith("x_")],
    }
