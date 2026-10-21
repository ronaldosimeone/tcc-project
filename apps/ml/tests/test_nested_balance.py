"""
Imbalance handling is a tuned hyper-parameter of the nested protocol.

1. Every search space keeps its a-priori default first and gains a denser
   normal-class sampling variant (and, except the autoencoder, a softer
   class-weight variant).
2. ``neg_stride`` controls how many normal rows a tabular learner sees.
3. ``pos_weight_scale`` scales the failure-class weight; the balancing keys
   never reach the learner's constructor.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

import src.nested_protocol as npc


def _ctx(n: int = 200, n_pos: int = 20) -> Any:
    rng = np.random.default_rng(0)
    y = np.zeros(n, int)
    y[:n_pos] = 1
    X = pd.DataFrame(rng.normal(size=(n, 3)) + y[:, None], columns=list("abc"))
    return SimpleNamespace(X=X, y=pd.Series(y))


def test_spaces_gain_balance_variants() -> None:
    for name, space in npc.SPACES.items():
        dense = [p for p in space if "neg_stride" in p]
        soft = [p for p in space if "pos_weight_scale" in p]
        assert dense and dense[0]["neg_stride"] == npc.DENSE_NEG_STRIDE[name]
        assert len(soft) == (0 if name == "autoencoder" else 1)
        assert "neg_stride" not in space[0] and "pos_weight_scale" not in space[0]


def test_neg_stride_controls_normal_rows(monkeypatch: Any) -> None:
    seen: list[int] = []

    class Spy:
        def fit(self, X: np.ndarray, y: np.ndarray) -> "Spy":
            seen.append(len(y))
            return self

    monkeypatch.setattr(npc, "_tree_model", lambda *a, **k: Spy())
    ctx = _ctx()
    idx = np.arange(200)
    npc.fit_model(ctx, "random_forest", idx, 0, {"neg_stride": 1})
    npc.fit_model(ctx, "random_forest", idx, 0, {"neg_stride": 5})
    assert seen == [200, 20 + 180 // 5]


def test_pos_weight_scale_reaches_xgboost_and_not_its_kwargs() -> None:
    model = npc._tree_model(  # noqa: SLF001
        "xgboost", {"max_depth": 2}, 0, pos_weight=40.0, scale=0.25
    )
    assert model.get_params()["scale_pos_weight"] == 10.0
    rf = npc._tree_model(
        "random_forest", {}, 0, pos_weight=40.0, scale=0.25
    )  # noqa: SLF001
    assert rf.get_params()["class_weight"] == {0: 1.0, 1: 10.0}
    rf_bal = npc._tree_model(
        "random_forest", {}, 0, pos_weight=40.0, scale=1.0
    )  # noqa: SLF001
    assert rf_bal.get_params()["class_weight"] == "balanced_subsample"
