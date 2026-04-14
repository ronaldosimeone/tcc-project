"""
Unit tests for apps/ml/src/balancing.py — MetroPTBalancer.

Mathematical guarantees verified
---------------------------------
1.  After SMOTE, minority class count >= majority class count * strategy
    (or == majority count when strategy="auto").
2.  Majority class count is NEVER altered by SMOTE (only minority is grown).
3.  Test split is NEVER touched by fit_resample.
4.  No synthetic rows appear in the test set.
5.  Total row count after resampling == sum of per-class counts after.
6.  Feature dimensionality is preserved exactly.
7.  train_test_split_safe preserves stratification ratios within 1 % tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.balancing import MetroPTBalancer, _class_counts, _read_sampling_strategy


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _imbalanced_df(
    n_majority: int = 500,
    n_minority: int = 50,
    n_features: int = 8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build a synthetic imbalanced dataset.
    Class 0 = majority (normal operation).
    Class 1 = minority (fault).
    """
    rng = np.random.default_rng(seed)
    X_maj = rng.normal(loc=0.0, scale=1.0, size=(n_majority, n_features))
    X_min = rng.normal(loc=3.0, scale=0.5, size=(n_minority, n_features))
    X = np.vstack([X_maj, X_min])
    y = np.array([0] * n_majority + [1] * n_minority)
    cols = [f"sensor_{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), pd.Series(y, name="fault")


# ---------------------------------------------------------------------------
# _read_sampling_strategy
# ---------------------------------------------------------------------------


class TestReadSamplingStrategy:
    def test_returns_auto_when_env_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("SMOTE_SAMPLING_STRATEGY", raising=False)
        assert _read_sampling_strategy() == "auto"

    def test_returns_float_when_valid_float_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SMOTE_SAMPLING_STRATEGY", "0.5")
        assert _read_sampling_strategy() == pytest.approx(0.5)

    def test_returns_keyword_minority(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SMOTE_SAMPLING_STRATEGY", "minority")
        assert _read_sampling_strategy() == "minority"

    def test_returns_auto_on_invalid_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SMOTE_SAMPLING_STRATEGY", "garbage_value")
        assert _read_sampling_strategy() == "auto"

    def test_returns_auto_on_empty_string(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SMOTE_SAMPLING_STRATEGY", "")
        assert _read_sampling_strategy() == "auto"

    def test_float_out_of_range_falls_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("SMOTE_SAMPLING_STRATEGY", "1.5")
        assert _read_sampling_strategy() == "auto"


# ---------------------------------------------------------------------------
# fit_resample — core mathematical guarantees
# ---------------------------------------------------------------------------


class TestFitResample:
    def test_returns_tuple_of_dataframe_and_series(self) -> None:
        X, y = _imbalanced_df()
        X_res, y_res = MetroPTBalancer().fit_resample(X, y)
        assert isinstance(X_res, pd.DataFrame)
        assert isinstance(y_res, pd.Series)

    def test_minority_class_grows(self) -> None:
        """After SMOTE(auto), minority count must equal majority count."""
        X, y = _imbalanced_df(n_majority=500, n_minority=50)
        X_res, y_res = MetroPTBalancer(sampling_strategy="auto").fit_resample(X, y)
        counts_after = _class_counts(y_res)
        assert (
            counts_after[1] == counts_after[0]
        ), f"Minority ({counts_after[1]}) should equal majority ({counts_after[0]}) after auto SMOTE."

    def test_majority_class_count_unchanged(self) -> None:
        """SMOTE must NEVER reduce or augment the majority class."""
        X, y = _imbalanced_df(n_majority=500, n_minority=50)
        counts_before = _class_counts(y)
        X_res, y_res = MetroPTBalancer(sampling_strategy="auto").fit_resample(X, y)
        counts_after = _class_counts(y_res)
        assert (
            counts_after[0] == counts_before[0]
        ), "Majority class count must not change."

    def test_total_row_count_is_consistent(self) -> None:
        """Total rows must equal sum of individual class counts."""
        X, y = _imbalanced_df()
        X_res, y_res = MetroPTBalancer().fit_resample(X, y)
        counts = _class_counts(y_res)
        assert len(X_res) == sum(counts.values())
        assert len(X_res) == len(y_res)

    def test_feature_dimensionality_preserved(self) -> None:
        """Synthetic rows must have identical column count to originals."""
        X, y = _imbalanced_df(n_features=8)
        X_res, _ = MetroPTBalancer().fit_resample(X, y)
        assert X_res.shape[1] == X.shape[1]

    def test_column_names_preserved(self) -> None:
        """DataFrame column names must survive resampling."""
        X, y = _imbalanced_df()
        X_res, _ = MetroPTBalancer().fit_resample(X, y)
        assert list(X_res.columns) == list(X.columns)

    def test_series_name_preserved(self) -> None:
        """Target Series name must be retained after resampling."""
        X, y = _imbalanced_df()
        _, y_res = MetroPTBalancer().fit_resample(X, y)
        assert y_res.name == "fault"

    def test_float_strategy_respects_ratio(self) -> None:
        """
        With strategy=0.5, minority/majority ratio after resampling must be >= 0.5.
        """
        X, y = _imbalanced_df(n_majority=500, n_minority=20)
        X_res, y_res = MetroPTBalancer(sampling_strategy=0.5).fit_resample(X, y)
        counts = _class_counts(y_res)
        ratio = counts[1] / counts[0]
        assert ratio >= 0.499, f"Expected ratio >= 0.5, got {ratio:.3f}"

    def test_accepts_numpy_arrays(self) -> None:
        """fit_resample must accept raw numpy arrays, not only DataFrames."""
        X, y = _imbalanced_df()
        X_res, y_res = MetroPTBalancer().fit_resample(X.values, y.values)
        assert isinstance(X_res, pd.DataFrame)
        assert isinstance(y_res, pd.Series)

    def test_borderline_smote_path(self) -> None:
        """borderline=True must execute without error and still balance classes."""
        X, y = _imbalanced_df(n_majority=300, n_minority=30)
        X_res, y_res = MetroPTBalancer(borderline=True).fit_resample(X, y)
        counts = _class_counts(y_res)
        assert counts[1] >= 30, "Minority must not shrink with BorderlineSMOTE."

    def test_reproducibility_with_fixed_seed(self) -> None:
        """Two calls with the same random_state must produce identical results."""
        X, y = _imbalanced_df()
        X1, y1 = MetroPTBalancer(random_state=0).fit_resample(X, y)
        X2, y2 = MetroPTBalancer(random_state=0).fit_resample(X, y)
        pd.testing.assert_frame_equal(X1, X2)
        pd.testing.assert_series_equal(y1, y2)

    def test_different_seeds_produce_different_synthetic_rows(self) -> None:
        """Different seeds must produce different synthetic samples."""
        X, y = _imbalanced_df()
        X1, _ = MetroPTBalancer(random_state=1).fit_resample(X, y)
        X2, _ = MetroPTBalancer(random_state=99).fit_resample(X, y)
        assert not X1.equals(X2), "Different seeds should yield different synthetics."


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    def test_raises_type_error_for_bad_X(self) -> None:
        _, y = _imbalanced_df()
        with pytest.raises(TypeError, match="DataFrame or ndarray"):
            MetroPTBalancer().fit_resample("not_valid", y)  # type: ignore[arg-type]

    def test_raises_type_error_for_bad_y(self) -> None:
        X, _ = _imbalanced_df()
        with pytest.raises(TypeError, match="Series or ndarray"):
            MetroPTBalancer().fit_resample(X, "not_valid")  # type: ignore[arg-type]

    def test_raises_value_error_when_lengths_differ(self) -> None:
        X, y = _imbalanced_df(n_majority=100, n_minority=10)
        with pytest.raises(ValueError, match="same length"):
            MetroPTBalancer().fit_resample(X, y.iloc[:50])

    def test_raises_value_error_for_single_class(self) -> None:
        X, y = _imbalanced_df()
        y_one_class = pd.Series(np.zeros(len(y), dtype=int), name="fault")
        with pytest.raises(ValueError, match="2 classes"):
            MetroPTBalancer().fit_resample(X, y_one_class)


# ---------------------------------------------------------------------------
# Anti-leakage: train_test_split_safe
# ---------------------------------------------------------------------------


class TestTrainTestSplitSafe:
    def test_returns_four_splits(self) -> None:
        X, y = _imbalanced_df()
        splits = MetroPTBalancer.train_test_split_safe(X, y, test_size=0.2)
        assert set(splits.keys()) == {"X_train", "X_test", "y_train", "y_test"}

    def test_split_sizes_are_correct(self) -> None:
        X, y = _imbalanced_df(n_majority=500, n_minority=50)
        n_total = len(X)
        splits = MetroPTBalancer.train_test_split_safe(X, y, test_size=0.2)
        assert len(splits["X_test"]) == pytest.approx(n_total * 0.2, abs=2)
        assert len(splits["X_train"]) == pytest.approx(n_total * 0.8, abs=2)

    def test_train_and_test_are_disjoint(self) -> None:
        """No index should appear in both train and test."""
        X, y = _imbalanced_df()
        splits = MetroPTBalancer.train_test_split_safe(X, y)
        train_idx = set(splits["X_train"].index)  # type: ignore[union-attr]
        test_idx = set(splits["X_test"].index)  # type: ignore[union-attr]
        assert train_idx.isdisjoint(test_idx), "Train and test sets overlap."

    def test_stratification_preserves_class_ratio(self) -> None:
        """
        Class ratio in train and test must be within 1 % of the full dataset ratio.
        """
        X, y = _imbalanced_df(n_majority=500, n_minority=50)
        overall_ratio: float = (y == 1).sum() / len(y)
        splits = MetroPTBalancer.train_test_split_safe(X, y, stratify=True)
        for key in ("y_train", "y_test"):
            split_y = splits[key]
            split_ratio = (split_y == 1).sum() / len(split_y)  # type: ignore[operator]
            assert (
                abs(split_ratio - overall_ratio) < 0.01
            ), f"{key} ratio {split_ratio:.3f} deviates > 1% from {overall_ratio:.3f}"

    def test_test_set_unchanged_after_resample(self) -> None:
        """
        The test set must be byte-identical before and after calling fit_resample
        on the training portion — proof that SMOTE never touches test data.
        """
        X, y = _imbalanced_df()
        splits = MetroPTBalancer.train_test_split_safe(X, y)

        X_test_before = splits["X_test"].copy()
        y_test_before = splits["y_test"].copy()

        # Resample training data
        MetroPTBalancer().fit_resample(splits["X_train"], splits["y_train"])

        pd.testing.assert_frame_equal(splits["X_test"], X_test_before)
        pd.testing.assert_series_equal(splits["y_test"], y_test_before)

    def test_no_synthetic_rows_in_test_set(self) -> None:
        """
        After resampling, every row in the test set must correspond to a row
        in the original dataset (index-based membership check).
        """
        X, y = _imbalanced_df()
        original_index = set(X.index.tolist())
        splits = MetroPTBalancer.train_test_split_safe(X, y)
        MetroPTBalancer().fit_resample(splits["X_train"], splits["y_train"])

        test_indices = set(splits["X_test"].index.tolist())  # type: ignore[union-attr]
        assert test_indices.issubset(original_index), (
            "Test set contains indices not in the original dataset — "
            "synthetic rows leaked into the test split."
        )

    def test_non_stratified_split_works(self) -> None:
        """stratify=False must not raise even on imbalanced data."""
        X, y = _imbalanced_df()
        splits = MetroPTBalancer.train_test_split_safe(X, y, stratify=False)
        assert len(splits["X_train"]) > 0


# ---------------------------------------------------------------------------
# Environment-driven strategy integration test
# ---------------------------------------------------------------------------


class TestEnvIntegration:
    def test_balancer_picks_up_env_strategy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """
        When SMOTE_SAMPLING_STRATEGY=0.8 is set, the balancer should produce
        a minority/majority ratio >= 0.8 after resampling.
        """
        monkeypatch.setenv("SMOTE_SAMPLING_STRATEGY", "0.8")
        X, y = _imbalanced_df(n_majority=500, n_minority=20)
        # sampling_strategy=None forces the env read path
        balancer = MetroPTBalancer(sampling_strategy=None)
        assert balancer.sampling_strategy == pytest.approx(0.8)
        X_res, y_res = balancer.fit_resample(X, y)
        counts = _class_counts(y_res)
        ratio = counts[1] / counts[0]
        assert ratio >= 0.799
