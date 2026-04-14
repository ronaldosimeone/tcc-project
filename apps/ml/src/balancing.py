"""
Class Balancing module — RF-03 / RNF-09.

Design decisions
----------------
1.  **SMOTE only on training data** (RF-03 hard rule).
    The public API is `fit_resample(X_train, y_train)` which operates
    exclusively on the training split.  There is no `transform` method that
    could accidentally be called on validation or test data.  The companion
    helper `train_test_split_safe` enforces the split *before* any resampling
    and returns the four canonical arrays (X_train, X_test, y_train, y_test).

2.  **Strategy read from environment** (RNF-09).
    `SMOTE_SAMPLING_STRATEGY` in `.env` accepts the same values as
    imbalanced-learn's `sampling_strategy` parameter: a float (target
    minority/majority ratio), ``"minority"``, ``"not majority"``, or ``"auto"``.
    When the variable is absent or unparseable, the module falls back to
    ``"auto"`` — imbalanced-learn's safest default (balances every class to
    match the majority count).

3.  **Isolation Forest / XGBoost compatibility**.
    SMOTE generates synthetic samples in feature space via k-NN interpolation.
    For tabular sensor data with correlated features this is superior to random
    over-sampling (which just duplicates rows) and avoids the information loss
    of under-sampling.  The default `k_neighbors=5` is kept; a separate
    `BorderlineSMOTE` path is exposed for cases where the fault boundary is
    tight (common in the MetroPT-3 air-pressure anomalies).

4.  **Stateless design**.
    `MetroPTBalancer` stores no fitted state between calls.  Each call to
    `fit_resample` creates a fresh SMOTE instance seeded by `random_state`,
    making the pipeline fully reproducible when the seed is fixed.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import BorderlineSMOTE, SMOTE  # type: ignore
from sklearn.model_selection import train_test_split

logger: logging.Logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants / environment
# ---------------------------------------------------------------------------

_ENV_KEY: str = "SMOTE_SAMPLING_STRATEGY"
_DEFAULT_STRATEGY: str | float = "auto"
_DEFAULT_K_NEIGHBORS: int = 5
_DEFAULT_RANDOM_STATE: int = 42


def _read_sampling_strategy() -> str | float:
    """
    Parse SMOTE_SAMPLING_STRATEGY from the environment.

    Accepted values
    ---------------
    - Float string, e.g. ``"0.5"``  →  converted to ``float``.
    - Keyword strings: ``"auto"``, ``"minority"``, ``"not minority"``,
      ``"not majority"``, ``"all"``.
    - Absent / empty  →  ``"auto"`` (safe fallback).
    """
    raw: str = os.getenv(_ENV_KEY, "").strip()

    if not raw:
        logger.debug(
            "%s not set – using default strategy '%s'.", _ENV_KEY, _DEFAULT_STRATEGY
        )
        return _DEFAULT_STRATEGY

    try:
        value: float = float(raw)
        if not (0.0 < value <= 1.0):
            raise ValueError(f"Float strategy must be in (0, 1], got {value}.")
        logger.info("SMOTE sampling strategy read from env: %.3f", value)
        return value
    except ValueError:
        pass

    valid_keywords: frozenset[str] = frozenset(
        {"auto", "minority", "not minority", "not majority", "all"}
    )
    if raw.lower() in valid_keywords:
        logger.info("SMOTE sampling strategy read from env: '%s'.", raw.lower())
        return raw.lower()

    logger.warning(
        "Invalid %s value '%s'. Falling back to '%s'.",
        _ENV_KEY,
        raw,
        _DEFAULT_STRATEGY,
    )
    return _DEFAULT_STRATEGY


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class MetroPTBalancer:
    """
    Applies SMOTE-based oversampling **exclusively to training splits**.

    Parameters
    ----------
    sampling_strategy : str | float | None
        Passed directly to ``SMOTE(sampling_strategy=...)``.
        ``None`` reads the value from the ``SMOTE_SAMPLING_STRATEGY``
        environment variable (or falls back to ``"auto"``).
    k_neighbors : int
        Number of nearest neighbours used by SMOTE's k-NN interpolation.
        Lower values (e.g. 3) are safer when the minority class is very small.
    random_state : int
        Seed for reproducibility across runs and cross-validation folds.
    borderline : bool
        When ``True``, uses ``BorderlineSMOTE`` instead of vanilla SMOTE.
        BorderlineSMOTE focuses synthetic generation near the decision boundary
        — recommended when the fault cluster is spatially tight in feature
        space (observed in MetroPT-3 valve-failure events).

    Examples
    --------
    >>> balancer = MetroPTBalancer()
    >>> X_res, y_res = balancer.fit_resample(X_train, y_train)

    Safe split + resample in one call:

    >>> splits = MetroPTBalancer.train_test_split_safe(X, y, test_size=0.2)
    >>> X_train_res, y_train_res = balancer.fit_resample(
    ...     splits["X_train"], splits["y_train"]
    ... )
    """

    def __init__(
        self,
        sampling_strategy: str | float | None = None,
        k_neighbors: int = _DEFAULT_K_NEIGHBORS,
        random_state: int = _DEFAULT_RANDOM_STATE,
        borderline: bool = False,
    ) -> None:
        self.sampling_strategy: str | float = (
            sampling_strategy
            if sampling_strategy is not None
            else _read_sampling_strategy()
        )
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.borderline = borderline

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def fit_resample(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Apply SMOTE oversampling and return the balanced training set.

        **NEVER call this method on validation or test data.**

        Parameters
        ----------
        X : pd.DataFrame | np.ndarray
            Feature matrix — must be the **training split only**.
        y : pd.Series | np.ndarray
            Target vector aligned with *X*.

        Returns
        -------
        X_resampled : pd.DataFrame
            Resampled feature matrix (original + synthetic rows).
            Column names are preserved when *X* is a DataFrame.
        y_resampled : pd.Series
            Resampled target vector aligned with *X_resampled*.
        """
        self._validate_inputs(X, y)

        original_counts: dict[Any, int] = _class_counts(y)
        logger.info("Class distribution BEFORE balancing: %s", original_counts)

        smote = self._build_smote()
        X_res_raw, y_res_raw = smote.fit_resample(X, y)  # type: ignore[arg-type]

        # Preserve DataFrame / Series types and column names
        X_resampled: pd.DataFrame = self._to_dataframe(X, X_res_raw)
        y_resampled: pd.Series = pd.Series(y_res_raw, name=_series_name(y))

        resampled_counts: dict[Any, int] = _class_counts(y_resampled)
        logger.info("Class distribution AFTER  balancing: %s", resampled_counts)
        self._log_report(original_counts, resampled_counts, len(X_resampled))

        return X_resampled, y_resampled

    # ------------------------------------------------------------------
    # Safe split helper (anti-leakage)
    # ------------------------------------------------------------------

    @staticmethod
    def train_test_split_safe(
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
        test_size: float = 0.2,
        random_state: int = _DEFAULT_RANDOM_STATE,
        stratify: bool = True,
    ) -> dict[str, pd.DataFrame | pd.Series | np.ndarray]:
        """
        Stratified train/test split with an explicit anti-leakage contract.

        The split is performed **before** any resampling.  This method is the
        recommended entry point for building the training pipeline:

        1. Call ``train_test_split_safe`` to obtain isolated splits.
        2. Call ``fit_resample`` on the training portion only.
        3. Fit the model on the resampled training set.
        4. Evaluate on the **untouched** test set.

        Parameters
        ----------
        stratify : bool
            When ``True`` (default), preserves class proportions in both
            splits — critical for severely imbalanced datasets so that the
            test set contains at least one minority sample per fold.

        Returns
        -------
        dict with keys: ``X_train``, ``X_test``, ``y_train``, ``y_test``.
        """
        stratify_arr: pd.Series | np.ndarray | None = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arr,
        )

        logger.info(
            "Split — train: %d rows | test: %d rows (test_size=%.0f%%, stratified=%s)",
            len(X_train),
            len(X_test),
            test_size * 100,
            stratify,
        )

        return {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_smote(self) -> SMOTE | BorderlineSMOTE:
        """Instantiate the correct SMOTE variant."""
        common: dict[str, Any] = {
            "sampling_strategy": self.sampling_strategy,
            "k_neighbors": self.k_neighbors,
            "random_state": self.random_state,
        }
        if self.borderline:
            return BorderlineSMOTE(**common)
        return SMOTE(**common)

    @staticmethod
    def _validate_inputs(
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> None:
        """Raise on obviously wrong inputs before touching SMOTE."""
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise TypeError(
                f"X must be a DataFrame or ndarray, got {type(X).__name__!r}."
            )
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise TypeError(f"y must be a Series or ndarray, got {type(y).__name__!r}.")
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length: {len(X)} vs {len(y)}."
            )
        n_classes: int = len(np.unique(y))
        if n_classes < 2:
            raise ValueError(f"SMOTE requires at least 2 classes, found {n_classes}.")

    @staticmethod
    def _to_dataframe(
        original: pd.DataFrame | np.ndarray,
        resampled_raw: np.ndarray,
    ) -> pd.DataFrame:
        """Wrap the raw numpy output back into a DataFrame with original columns."""
        if isinstance(original, pd.DataFrame):
            return pd.DataFrame(resampled_raw, columns=original.columns)
        return pd.DataFrame(resampled_raw)

    @staticmethod
    def _log_report(
        before: dict[Any, int],
        after: dict[Any, int],
        total_after: int,
    ) -> None:
        """Emit a structured summary to stdout / log."""
        lines: list[str] = [
            "",
            "┌─────────────────────────────────────────────────┐",
            "│         SMOTE Resampling Report                 │",
            "├──────────┬────────────┬────────────┬────────────┤",
            "│  Class   │   Before   │   After    │  Synthetic │",
            "├──────────┼────────────┼────────────┼────────────┤",
        ]
        for cls in sorted(after.keys()):
            n_before: int = before.get(cls, 0)
            n_after: int = after[cls]
            synthetic: int = n_after - n_before
            lines.append(
                f"│  {str(cls):<8}│ {n_before:>10} │ {n_after:>10} │ {synthetic:>10} │"
            )
        lines += [
            "├──────────┴────────────┴────────────┴────────────┤",
            f"│  Total after resampling: {total_after:>22} │",
            "└─────────────────────────────────────────────────┘",
        ]
        logger.info("\n".join(lines))


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _class_counts(y: pd.Series | np.ndarray) -> dict[Any, int]:
    """Return {class_label: count} for all classes in *y*."""
    values, counts = np.unique(y, return_counts=True)
    return dict(zip(values.tolist(), counts.tolist()))


def _series_name(y: pd.Series | np.ndarray) -> str:
    """Preserve Series name or fall back to 'target'."""
    if isinstance(y, pd.Series) and y.name:
        return str(y.name)
    return "target"
