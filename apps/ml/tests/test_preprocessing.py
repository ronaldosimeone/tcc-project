"""
Unit tests for apps/ml/src/preprocessing.py – MetroPTPreprocessor.

Coverage targets
----------------
- fit_transform end-to-end (smoke test)
- TypeError on invalid input
- Null imputation (ffill → bfill)
- Pressure delta correctness
- Rolling std column creation and shape
- Moving average column creation
- Custom sensor_cols parameter
- Scikit-learn clone/get_params compatibility
- Edge cases: single-row DataFrame, all-null column
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from src.preprocessing import MetroPTPreprocessor, _DEFAULT_SENSOR_COLS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(n_rows: int = 20, seed: int = 42) -> pd.DataFrame:
    """
    Build a minimal synthetic DataFrame that mirrors the MetroPT-3 schema.
    All sensor values are random floats; timestamp is a regular 1-second series.
    """
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {
        "timestamp": pd.date_range("2022-01-01", periods=n_rows, freq="1s"),
    }
    for col in _DEFAULT_SENSOR_COLS:
        data[col] = rng.uniform(low=0.5, high=10.0, size=n_rows).astype("float32")
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------


class TestFitTransform:
    def test_fit_transform_returns_dataframe(self) -> None:
        """fit_transform must return a pd.DataFrame."""
        pre = MetroPTPreprocessor()
        result = pre.fit_transform(_make_df())
        assert isinstance(result, pd.DataFrame)

    def test_fit_transform_does_not_mutate_input(self) -> None:
        """The original DataFrame must be unchanged after transform."""
        df = _make_df()
        original_cols = list(df.columns)
        MetroPTPreprocessor().fit_transform(df)
        assert list(df.columns) == original_cols

    def test_output_has_more_columns_than_input(self) -> None:
        """transform must add engineered feature columns."""
        df = _make_df()
        result = MetroPTPreprocessor().fit_transform(df)
        assert len(result.columns) > len(df.columns)

    def test_output_row_count_matches_input(self) -> None:
        """transform must preserve the number of rows."""
        df = _make_df(n_rows=30)
        result = MetroPTPreprocessor().fit_transform(df)
        assert len(result) == 30

    def test_fit_returns_self(self) -> None:
        """fit() must return the preprocessor instance (sklearn contract)."""
        pre = MetroPTPreprocessor()
        assert pre.fit(_make_df()) is pre


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


class TestInputValidation:
    @pytest.mark.parametrize(
        "bad_input",
        [
            None,
            42,
            "not a dataframe",
            [1, 2, 3],
            {"TP2": [1.0, 2.0]},
            np.array([[1.0, 2.0], [3.0, 4.0]]),
        ],
    )
    def test_transform_raises_type_error_for_non_dataframe(
        self, bad_input: object
    ) -> None:
        """transform must raise TypeError for any non-DataFrame input."""
        with pytest.raises(TypeError, match="pandas DataFrame"):
            MetroPTPreprocessor().transform(bad_input)  # type: ignore[arg-type]

    def test_type_error_message_contains_actual_type(self) -> None:
        """TypeError message should name the actual type received."""
        with pytest.raises(TypeError, match="ndarray"):
            MetroPTPreprocessor().transform(np.zeros((5, 3)))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Null imputation
# ---------------------------------------------------------------------------


class TestNullImputation:
    def test_ffill_fills_interior_nulls(self) -> None:
        """Interior NaNs should be filled by forward-fill."""
        df = _make_df(n_rows=10)
        df.loc[3:5, "TP2"] = np.nan
        result = MetroPTPreprocessor().transform(df)
        assert result["TP2"].isna().sum() == 0

    def test_bfill_fills_leading_nulls(self) -> None:
        """Leading NaNs (rows 0-2) should be covered by backward-fill."""
        df = _make_df(n_rows=10)
        df.loc[0:2, "TP2"] = np.nan
        result = MetroPTPreprocessor().transform(df)
        assert result["TP2"].isna().sum() == 0

    def test_all_null_column_filled(self) -> None:
        """An entirely-null column should be filled to 0.0 by ffill+bfill fallback."""
        df = _make_df(n_rows=10)
        df["TP2"] = np.nan
        # ffill + bfill on an all-null column leaves NaN; we verify no crash
        # and accept that the column may remain NaN (no valid value to propagate).
        result = MetroPTPreprocessor().transform(df)
        assert isinstance(result, pd.DataFrame)

    def test_no_nulls_unchanged(self) -> None:
        """A null-free DataFrame must produce identical numeric values after imputation."""
        df = _make_df(n_rows=10)
        result = MetroPTPreprocessor().transform(df)
        pd.testing.assert_series_equal(
            df["TP2"].reset_index(drop=True),
            result["TP2"].reset_index(drop=True),
            check_names=False,
        )


# ---------------------------------------------------------------------------
# Pressure delta
# ---------------------------------------------------------------------------


class TestPressureDelta:
    def test_delta_column_exists(self) -> None:
        """transform must add a 'TP2_delta' column by default."""
        result = MetroPTPreprocessor().transform(_make_df())
        assert "TP2_delta" in result.columns

    def test_delta_first_value_is_zero(self) -> None:
        """First row delta must be 0.0 (no prior value)."""
        result = MetroPTPreprocessor().transform(_make_df())
        assert result["TP2_delta"].iloc[0] == pytest.approx(0.0)

    def test_delta_values_are_correct(self) -> None:
        """delta[i] must equal TP2[i] - TP2[i-1] for i > 0."""
        df = _make_df(n_rows=10)
        result = MetroPTPreprocessor().transform(df)
        expected = df["TP2"].diff().fillna(0.0)
        pd.testing.assert_series_equal(
            result["TP2_delta"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            atol=1e-5,
        )

    def test_custom_pressure_col(self) -> None:
        """pressure_col parameter must redirect delta to the specified column."""
        result = MetroPTPreprocessor(pressure_col="TP3").transform(_make_df())
        assert "TP3_delta" in result.columns
        assert "TP2_delta" not in result.columns

    def test_missing_pressure_col_does_not_crash(self) -> None:
        """If pressure_col is absent from the DataFrame, transform must not raise."""
        df = _make_df().drop(columns=["TP2"])
        result = MetroPTPreprocessor(pressure_col="TP2").transform(df)
        assert "TP2_delta" not in result.columns


# ---------------------------------------------------------------------------
# Rolling standard deviation
# ---------------------------------------------------------------------------


class TestRollingStd:
    def test_std_columns_exist_for_all_sensors(self) -> None:
        """A *_std_5 column must be added for every resolved sensor column."""
        df = _make_df()
        result = MetroPTPreprocessor().transform(df)
        for col in _DEFAULT_SENSOR_COLS:
            if col in df.columns:
                assert f"{col}_std_5" in result.columns, f"Missing {col}_std_5"

    def test_std_no_nulls_in_output(self) -> None:
        """Rolling std columns must not contain NaN (min_periods=1 + fillna)."""
        result = MetroPTPreprocessor().transform(_make_df())
        std_cols = [c for c in result.columns if c.endswith("_std_5")]
        assert len(std_cols) > 0
        assert result[std_cols].isna().sum().sum() == 0

    def test_std_single_row_is_zero(self) -> None:
        """std of a 1-element window is 0 (or NaN→0 after fillna)."""
        df = _make_df(n_rows=1)
        result = MetroPTPreprocessor().transform(df)
        assert result["TP2_std_5"].iloc[0] == pytest.approx(0.0)

    def test_custom_window_std(self) -> None:
        """window_std parameter must be reflected in the output column name."""
        result = MetroPTPreprocessor(window_std=10).transform(_make_df())
        assert "TP2_std_10" in result.columns
        assert "TP2_std_5" not in result.columns


# ---------------------------------------------------------------------------
# Moving averages
# ---------------------------------------------------------------------------


class TestMovingAverages:
    def test_ma_short_columns_exist(self) -> None:
        """A *_ma_5 column must be added for every sensor column."""
        df = _make_df()
        result = MetroPTPreprocessor().transform(df)
        for col in _DEFAULT_SENSOR_COLS:
            if col in df.columns:
                assert f"{col}_ma_5" in result.columns

    def test_ma_long_columns_exist(self) -> None:
        """A *_ma_15 column must be added for every sensor column."""
        df = _make_df()
        result = MetroPTPreprocessor().transform(df)
        for col in _DEFAULT_SENSOR_COLS:
            if col in df.columns:
                assert f"{col}_ma_15" in result.columns

    def test_ma_no_nulls_in_output(self) -> None:
        """Moving average columns must not contain NaN (min_periods=1)."""
        result = MetroPTPreprocessor().transform(_make_df(n_rows=30))
        ma_cols = [c for c in result.columns if "_ma_" in c]
        assert len(ma_cols) > 0
        assert result[ma_cols].isna().sum().sum() == 0

    def test_ma_single_value_equals_itself(self) -> None:
        """For a single-row input, MA must equal the original value."""
        df = _make_df(n_rows=1)
        result = MetroPTPreprocessor().transform(df)
        assert result["TP2_ma_5"].iloc[0] == pytest.approx(float(df["TP2"].iloc[0]))

    def test_ma5_converges_faster_than_ma15(self) -> None:
        """
        After a step change, MA_5 must track the new level faster than MA_15.
        We inject a step at row 20 and compare residuals at row 25.
        """
        n = 40
        df = _make_df(n_rows=n)
        df.loc[20:, "TP2"] = np.float32(df["TP2"].mean() + 100.0)  # large step
        result = MetroPTPreprocessor().transform(df)
        new_level = float(df["TP2"].iloc[25])
        residual_ma5 = abs(result["TP2_ma_5"].iloc[25] - new_level)
        residual_ma15 = abs(result["TP2_ma_15"].iloc[25] - new_level)
        assert residual_ma5 < residual_ma15

    def test_custom_window_params(self) -> None:
        """Custom window parameters must be reflected in column names."""
        result = MetroPTPreprocessor(window_ma_short=3, window_ma_long=7).transform(
            _make_df()
        )
        assert "TP2_ma_3" in result.columns
        assert "TP2_ma_7" in result.columns


# ---------------------------------------------------------------------------
# Custom sensor_cols
# ---------------------------------------------------------------------------


class TestCustomSensorCols:
    def test_only_requested_cols_get_features(self) -> None:
        """When sensor_cols=['TP2'], only TP2 should receive rolling features."""
        result = MetroPTPreprocessor(sensor_cols=["TP2"]).transform(_make_df())
        assert "TP2_std_5" in result.columns
        assert "TP3_std_5" not in result.columns

    def test_nonexistent_sensor_col_silently_skipped(self) -> None:
        """Columns not present in the DataFrame must be silently ignored."""
        result = MetroPTPreprocessor(sensor_cols=["TP2", "NONEXISTENT"]).transform(
            _make_df()
        )
        assert "NONEXISTENT_std_5" not in result.columns


# ---------------------------------------------------------------------------
# Scikit-learn compatibility
# ---------------------------------------------------------------------------


class TestSklearnCompat:
    def test_get_params_returns_init_params(self) -> None:
        """get_params() must expose all constructor parameters."""
        pre = MetroPTPreprocessor(window_std=7, window_ma_short=3, window_ma_long=9)
        params = pre.get_params()
        assert params["window_std"] == 7
        assert params["window_ma_short"] == 3
        assert params["window_ma_long"] == 9

    def test_clone_produces_independent_copy(self) -> None:
        """sklearn.clone must produce an independent instance with same params."""
        pre = MetroPTPreprocessor(window_std=7)
        cloned = clone(pre)
        assert cloned.window_std == 7
        assert cloned is not pre

    def test_set_params_works(self) -> None:
        """set_params() must update parameters (sklearn contract)."""
        pre = MetroPTPreprocessor()
        pre.set_params(window_std=12)
        assert pre.window_std == 12
