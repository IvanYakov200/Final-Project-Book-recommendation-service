"""Продвинутая часть. Нейросетевой ранжировщик (MLP) на PyTorch.

Вход: те же признаки пары (user, book), что у бустинга (скоры моделей + признаки пользователя, книги
и взаимодействия). Выход: скор релевантности. Два варианта обучения:

- loss="bce"      : pointwise, бинарная кросс-энтропия «релевантна / нет» (с весом положительных);
- loss="pairwise" : BPR-style, для каждой релевантной книги пользователя сэмплируются нерелевантные
                    книги ТОГО ЖЕ пользователя, и модель учится давать релевантной скор выше:
                    L = -log σ(s_pos - s_neg). Это прямо оптимизирует порядок внутри выдачи, а не
                    абсолютную вероятность, что ближе к nDCG.

Ранняя остановка по nDCG@10 на валидационном пуле (tune_users): сохраняем веса лучшей эпохи.
Стандартизация признаков + клиппинг в [-5, 5]: у z-скоров моделей тяжёлые хвосты (до 30–100σ),
и без клиппинга они доминируют в первых слоях.
"""

from __future__ import annotations

import copy
import time

import numpy as np
import pandas as pd


class MLP:
    """Небольшая обёртка над torch.nn.Sequential, чтобы torch импортировался лениво."""

    def __init__(self, n_in, hidden=(128, 64), dropout=0.2):
        import torch.nn as nn
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)


class NeuralRanker:
    def __init__(self, feature_cols, hidden=(128, 64), dropout=0.2, epochs=15, lr=1e-3,
                 batch_size=2048, weight_decay=1e-5, loss="pairwise", n_neg=8, seed=42, n_threads=4):
        self.feature_cols = list(feature_cols)
        self.hidden, self.dropout = tuple(hidden), dropout
        self.epochs, self.lr, self.batch_size, self.weight_decay = epochs, lr, batch_size, weight_decay
        self.loss_kind, self.n_neg, self.seed, self.n_threads = loss, n_neg, seed, n_threads
        self.mean_ = self.std_ = None
        self.state_dict_ = None
        self.history = []
        self.name = f"nn_{loss}"

    # ---------- данные ----------
    def _prep(self, pool: pd.DataFrame) -> np.ndarray:
        X = pool[self.feature_cols].to_numpy(dtype=np.float32)
        X = (X - self.mean_) / self.std_
        return np.clip(X, -5, 5).astype(np.float32)

    def _build(self):
        import torch
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.n_threads)
        return MLP(len(self.feature_cols), self.hidden, self.dropout).net

    def _pairs(self, pool: pd.DataFrame, rng: np.random.Generator):
        """Индексы (pos, neg) внутри одного пользователя: для каждой релевантной книги n_neg нерелевантных."""
        u = pool["user_id"].to_numpy(); y = pool["label"].to_numpy()
        order = np.argsort(u, kind="stable")
        u_s, y_s = u[order], y[order]
        bounds = np.flatnonzero(np.r_[True, u_s[1:] != u_s[:-1], True])
        pos_idx, neg_idx = [], []
        for a, b in zip(bounds[:-1], bounds[1:]):
            seg = order[a:b]; ys = y_s[a:b]
            P, N = seg[ys == 1], seg[ys == 0]
            if len(P) == 0 or len(N) == 0:
                continue
            pos_idx.append(np.repeat(P, self.n_neg))
            neg_idx.append(rng.choice(N, size=len(P) * self.n_neg, replace=True))
        return np.concatenate(pos_idx), np.concatenate(neg_idx)

    # ---------- обучение ----------
    def fit(self, pool: pd.DataFrame, val_pool: pd.DataFrame | None = None, ds=None, verbose: bool = True):
        import torch
        import torch.nn.functional as F
        from .hybrid import mean_ndcg

        t0 = time.time()
        Xraw = pool[self.feature_cols].to_numpy(dtype=np.float32)
        self.mean_ = Xraw.mean(axis=0); self.std_ = Xraw.std(axis=0) + 1e-6
        X = torch.from_numpy(self._prep(pool)); y = torch.from_numpy(pool["label"].to_numpy(dtype=np.float32))
        net = self._build()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        rng = np.random.default_rng(self.seed)
        pos_weight = torch.tensor([(1 - y.mean()) / y.mean()]) if self.loss_kind == "bce" else None

        best, best_state, self.history = -1.0, None, []
        for epoch in range(1, self.epochs + 1):
            net.train(); total, nb = 0.0, 0
            if self.loss_kind == "bce":
                perm = torch.from_numpy(rng.permutation(len(X)))
                for i in range(0, len(X), self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    logits = net(X[idx]).squeeze(1)
                    loss = F.binary_cross_entropy_with_logits(logits, y[idx], pos_weight=pos_weight)
                    opt.zero_grad(); loss.backward(); opt.step()
                    total += loss.item(); nb += 1
            else:
                P, N = self._pairs(pool, rng)
                perm = rng.permutation(len(P)); P, N = torch.from_numpy(P[perm]), torch.from_numpy(N[perm])
                for i in range(0, len(P), self.batch_size):
                    sp = net(X[P[i:i + self.batch_size]]).squeeze(1)
                    sn = net(X[N[i:i + self.batch_size]]).squeeze(1)
                    loss = F.softplus(-(sp - sn)).mean()          # -log σ(s_pos − s_neg)
                    opt.zero_grad(); loss.backward(); opt.step()
                    total += loss.item(); nb += 1
            rec = {"epoch": epoch, "train_loss": total / max(nb, 1)}
            if val_pool is not None and ds is not None:
                rec["val_ndcg@10"] = mean_ndcg(val_pool, self._score_with(net, val_pool), ds)
                if rec["val_ndcg@10"] > best:
                    best, best_state = rec["val_ndcg@10"], copy.deepcopy(net.state_dict())
            self.history.append(rec)
            if verbose:
                msg = f"  epoch {epoch:2d}  loss={rec['train_loss']:.4f}"
                if "val_ndcg@10" in rec:
                    msg += f"  val nDCG@10={rec['val_ndcg@10']:.4f}" + (" *" if rec["val_ndcg@10"] == best else "")
                print(msg, flush=True)
        self.state_dict_ = best_state if best_state is not None else copy.deepcopy(net.state_dict())
        self._net = None
        if verbose:
            print(f"  done in {time.time() - t0:.0f}s, best val nDCG@10 = {best:.4f}")
        return self

    # ---------- предсказание ----------
    def _get_net(self):
        net = getattr(self, "_net", None)
        if net is None:
            net = self._build(); net.load_state_dict(self.state_dict_); net.eval()
            self._net = net
        return net

    def _score_with(self, net, pool: pd.DataFrame) -> np.ndarray:
        import torch
        was_training = net.training
        net.eval()
        X = torch.from_numpy(self._prep(pool))
        with torch.no_grad():
            out = [net(X[i:i + 65536]).squeeze(1).numpy() for i in range(0, len(X), 65536)]
        if was_training:
            net.train()
        return np.concatenate(out).astype(float) if out else np.zeros(0)

    def score(self, pool: pd.DataFrame) -> np.ndarray:
        return self._score_with(self._get_net(), pool)

    def __getstate__(self):
        d = self.__dict__.copy(); d["_net"] = None
        return d


class RankAverage:
    """Интеграция нейросети в гибрид: усреднение рангов (внутри пользователя) нескольких ранжировщиков."""

    def __init__(self, scorers: dict, weights: dict | None = None):
        self.scorers = scorers
        self.weights = weights or {k: 1.0 for k in scorers}

    def score(self, pool: pd.DataFrame) -> np.ndarray:
        u = pool["user_id"].to_numpy()
        out = np.zeros(len(pool))
        for name, sc in self.scorers.items():
            s = sc.score(pool) if hasattr(sc, "score") else sc(pool)
            # ранг внутри пользователя, нормированный в [0, 1] (1 = лучший)
            r = pd.Series(s).groupby(u).rank(pct=True).to_numpy()
            out += self.weights[name] * r
        return out / sum(self.weights.values())
