"""
Testes de `MetroPTPreprocessor` (feature engineering, RNF-64/RNF-65).

Auditoria desta task: `src/services/preprocessing.py` tinha 28% de cobertura
e ZERO teste dedicado — só era exercitado de raspão via `InferencePipelineService`
quando o buffer estava aquecido. Nenhum dos métodos privados (`_impute_nulls`,
`_add_pressure_delta`, `_add_rolling_std`, `_add_moving_averages`,
`_add_cross_sensor_features`, `_add_lags`, `_add_rolling_minmax`) tinha
verificação direta do RESULTADO numérico — exatamente o tipo de lacuna que o
mutation testing (RNF-64) expõe (troca de operador `-`↔`+`, `/`↔`*`, off-by-one
em janela, `issubset`↔sempre-verdadeiro etc. não seriam detectadas).

Cada teste abaixo verifica um VALOR calculado específico (não só "não lançou
exceção" nem "coluna existe") contra um DataFrame pequeno e determinístico.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.services.preprocessing import (
    MetroPTPreprocessor,
    _DEFAULT_SENSOR_COLS,
    _RATIO_EPS,
)


# ---------------------------------------------------------------------------
# Constantes e defaults do construtor (RNF-64) — nenhum teste pré-existente
# construía `MetroPTPreprocessor()` sem sobrescrever TODOS os parâmetros
# numéricos, então o valor DEFAULT de cada um nunca era verificado.
# ---------------------------------------------------------------------------


def test_default_sensor_cols_is_the_exact_expected_list() -> None:
    assert _DEFAULT_SENSOR_COLS == [
        "TP2",
        "TP3",
        "H1",
        "DV_pressure",
        "Reservoirs",
        "Oil_temperature",
        "Motor_current",
    ]


def test_constructor_defaults_produce_the_documented_window_and_lag_sizes() -> None:
    pre = MetroPTPreprocessor()
    assert pre.window_std == 5
    assert pre.window_ma_short == 5
    assert pre.window_ma_long == 15
    assert pre.enable_v2_features is True
    assert pre.lag_short == 5
    assert pre.lag_long == 15
    assert pre.window_minmax == 15


def test_default_window_ma_short_and_long_produce_the_expected_column_names() -> None:
    """Prova via comportamento observável (não só o atributo cru): com os
    defaults reais, `transform()` deve produzir `TP2_ma_5`/`TP2_ma_15` —
    um mutante que mudasse o default pra 6/16 produziria
    `TP2_ma_6`/`TP2_ma_16` em vez disso, e as colunas esperadas sumiriam."""
    pre = MetroPTPreprocessor(sensor_cols=["TP2"])
    df = pd.DataFrame({"TP2": [1.0, 2.0, 3.0]})
    out = pre.transform(df)
    assert "TP2_ma_5" in out.columns
    assert "TP2_ma_15" in out.columns
    assert "TP2_lag_5" in out.columns
    assert "TP2_lag_15" in out.columns
    assert "TP2_min_15" in out.columns
    assert "TP2_max_15" in out.columns


def test_enable_v2_features_defaults_to_true() -> None:
    """Sem passar `enable_v2_features`, as colunas V2 (lag/minmax/cross)
    devem existir — o default real é `True`, não `False`."""
    pre = MetroPTPreprocessor(sensor_cols=["TP2"])
    out = pre.transform(pd.DataFrame({"TP2": [1.0, 2.0, 3.0]}))
    assert "TP2_lag_5" in out.columns
    assert "TP2_min_15" in out.columns


# ---------------------------------------------------------------------------
# transform() — contrato geral
# ---------------------------------------------------------------------------


def test_transform_rejects_non_dataframe_input() -> None:
    pre = MetroPTPreprocessor()
    with pytest.raises(TypeError, match="pandas DataFrame"):
        pre.transform([1, 2, 3])  # type: ignore[arg-type]


def test_fit_returns_self_and_is_a_noop() -> None:
    pre = MetroPTPreprocessor()
    df = pd.DataFrame({"TP2": [1.0, 2.0]})
    assert pre.fit(df) is pre
    # fit não deve mutar o próprio df de entrada (stateless, sklearn contract).
    pd.testing.assert_frame_equal(df, pd.DataFrame({"TP2": [1.0, 2.0]}))


def test_transform_does_not_mutate_input_dataframe() -> None:
    """`transform` faz `X.copy()` antes de tudo — o df original do chamador
    nunca deve ser alterado (efeito colateral silencioso seria um bug real)."""
    pre = MetroPTPreprocessor(window_std=2, window_ma_short=2, window_ma_long=3)
    original = pd.DataFrame({"TP2": [1.0, 2.0, 3.0, 4.0]})
    snapshot = original.copy()

    pre.transform(original)

    pd.testing.assert_frame_equal(original, snapshot)


# ---------------------------------------------------------------------------
# _impute_nulls — ffill + bfill
# ---------------------------------------------------------------------------


def test_impute_nulls_forward_fills_interior_gap() -> None:
    pre = MetroPTPreprocessor()
    df = pd.DataFrame({"TP2": [1.0, np.nan, np.nan, 4.0]})
    out = pre._impute_nulls(df)
    assert out["TP2"].tolist() == [1.0, 1.0, 1.0, 4.0]


def test_impute_nulls_backward_fills_leading_gap() -> None:
    """Um NaN no INÍCIO da série não tem valor anterior para `ffill` —
    só `bfill` (aplicado depois) resolve. Sem o bfill, a linha 0
    continuaria NaN e quebraria tudo a jusante (rolling/lag)."""
    pre = MetroPTPreprocessor()
    df = pd.DataFrame({"TP2": [np.nan, 2.0, 3.0]})
    out = pre._impute_nulls(df)
    assert out["TP2"].tolist() == [2.0, 2.0, 3.0]


def test_impute_nulls_ignores_non_numeric_columns() -> None:
    pre = MetroPTPreprocessor()
    df = pd.DataFrame({"TP2": [1.0, np.nan], "label": ["a", None]})
    out = pre._impute_nulls(df)
    # Coluna não-numérica não é tocada pelo imputer (select_dtypes(number)) —
    # o nulo original continua lá (pandas normaliza `None` em coluna object
    # para NaN na leitura, por isso o check é `isna`, não igualdade a `None`).
    assert out["label"].iloc[0] == "a"
    assert pd.isna(out["label"].iloc[1])


# ---------------------------------------------------------------------------
# _add_pressure_delta
# ---------------------------------------------------------------------------


def test_pressure_delta_is_first_difference() -> None:
    pre = MetroPTPreprocessor(pressure_col="TP2")
    df = pd.DataFrame({"TP2": [10.0, 12.0, 9.0]})
    out = pre._add_pressure_delta(df)
    # diff(): NaN->0 na primeira linha, depois delta real.
    assert out["TP2_delta"].tolist() == [0.0, 2.0, -3.0]


def test_pressure_delta_noop_when_pressure_col_absent() -> None:
    pre = MetroPTPreprocessor(pressure_col="TP2")
    df = pd.DataFrame({"Other": [1.0, 2.0]})
    out = pre._add_pressure_delta(df)
    assert "TP2_delta" not in out.columns
    assert list(out.columns) == ["Other"]


# ---------------------------------------------------------------------------
# _resolve_sensor_cols
# ---------------------------------------------------------------------------


def test_resolve_sensor_cols_uses_default_list_when_none_configured() -> None:
    pre = MetroPTPreprocessor(sensor_cols=None)
    df = pd.DataFrame({"TP2": [1.0], "TP3": [2.0], "unrelated": [9.0]})
    assert pre._resolve_sensor_cols(df) == ["TP2", "TP3"]


def test_resolve_sensor_cols_uses_custom_list_when_configured() -> None:
    pre = MetroPTPreprocessor(sensor_cols=["Oil_temperature"])
    df = pd.DataFrame({"TP2": [1.0], "Oil_temperature": [70.0]})
    assert pre._resolve_sensor_cols(df) == ["Oil_temperature"]


def test_resolve_sensor_cols_drops_columns_missing_from_dataframe() -> None:
    pre = MetroPTPreprocessor(sensor_cols=["TP2", "DoesNotExist"])
    df = pd.DataFrame({"TP2": [1.0]})
    assert pre._resolve_sensor_cols(df) == ["TP2"]


# ---------------------------------------------------------------------------
# _add_rolling_std
# ---------------------------------------------------------------------------


def test_rolling_std_matches_pandas_rolling_std_with_min_periods_one() -> None:
    pre = MetroPTPreprocessor(sensor_cols=["TP2"], window_std=3)
    values = [1.0, 2.0, 3.0, 10.0]
    df = pd.DataFrame({"TP2": values})
    out = pre._add_rolling_std(df)

    expected = pd.Series(values).rolling(window=3, min_periods=1).std().fillna(0.0)
    assert out["TP2_std_3"].tolist() == pytest.approx(expected.tolist())
    # Primeira linha: janela de 1 amostra -> std indefinido -> preenchido com 0,
    # nunca NaN (um NaN vazando pro modelo quebraria a inferência ONNX).
    assert out["TP2_std_3"].iloc[0] == 0.0


def test_rolling_std_column_name_encodes_the_window_size() -> None:
    pre = MetroPTPreprocessor(sensor_cols=["TP2"], window_std=7)
    out = pre._add_rolling_std(pd.DataFrame({"TP2": [1.0, 2.0]}))
    assert "TP2_std_7" in out.columns


# ---------------------------------------------------------------------------
# _add_moving_averages
# ---------------------------------------------------------------------------


def test_moving_averages_short_and_long_windows_both_computed() -> None:
    pre = MetroPTPreprocessor(sensor_cols=["TP2"], window_ma_short=2, window_ma_long=4)
    df = pd.DataFrame({"TP2": [1.0, 2.0, 3.0, 4.0]})
    out = pre._add_moving_averages(df)

    # MA(2) na última linha: mean(3, 4) = 3.5
    assert out["TP2_ma_2"].iloc[-1] == pytest.approx(3.5)
    # MA(4) na última linha: mean(1,2,3,4) = 2.5
    assert out["TP2_ma_4"].iloc[-1] == pytest.approx(2.5)
    # min_periods=1 -> primeira linha é a própria amostra, não NaN.
    assert out["TP2_ma_2"].iloc[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# _add_cross_sensor_features — v2
# ---------------------------------------------------------------------------


def test_cross_sensor_features_all_present_when_columns_available() -> None:
    pre = MetroPTPreprocessor()
    df = pd.DataFrame(
        {
            "TP2": [10.0],
            "TP3": [4.0],
            "Motor_current": [2.0],
            "Reservoirs": [7.0],
        }
    )
    out = pre._add_cross_sensor_features(df)

    assert out["TP2_TP3_diff"].iloc[0] == pytest.approx(6.0)
    # RNF-64: valor esperado com o literal 1e-6 HARD-CODED, não `_RATIO_EPS`
    # importado — um teste que reusa a MESMA constante do módulo sob teste
    # nunca detecta uma mutação nela (o valor esperado mudaria junto com o
    # valor real, sob o mesmo processo mutado).
    assert out["TP2_TP3_ratio"].iloc[0] == pytest.approx(10.0 / (4.0 + 1e-6))
    assert out["work_per_pressure"].iloc[0] == pytest.approx(2.0 / (10.0 + 1e-6))
    assert out["reservoir_drop"].iloc[0] == pytest.approx(3.0)


def test_cross_sensor_ratio_never_divides_by_exact_zero() -> None:
    """TP3=0 é um valor real e válido do sensor — a razão usa +_RATIO_EPS
    exatamente para nunca produzir ZeroDivisionError/inf nesse caso."""
    pre = MetroPTPreprocessor()
    df = pd.DataFrame({"TP2": [5.0], "TP3": [0.0]})
    out = pre._add_cross_sensor_features(df)
    assert np.isfinite(out["TP2_TP3_ratio"].iloc[0])
    assert out["TP2_TP3_ratio"].iloc[0] == pytest.approx(5.0 / 1e-6)


def test_work_per_pressure_never_divides_by_exact_zero_and_uses_plus_eps() -> None:
    """Mesma fronteira que o teste acima, mas para `work_per_pressure`
    (TP2 == 0) — mata especificamente o mutante que troca `+_RATIO_EPS`
    por `-_RATIO_EPS` nessa fórmula (indistinguível quando TP2 não é
    próximo de zero)."""
    pre = MetroPTPreprocessor()
    df = pd.DataFrame({"TP2": [0.0], "Motor_current": [3.0]})
    out = pre._add_cross_sensor_features(df)
    assert np.isfinite(out["work_per_pressure"].iloc[0])
    assert out["work_per_pressure"].iloc[0] == pytest.approx(3.0 / 1e-6)


def test_ratio_eps_constant_is_exactly_1e_minus_6() -> None:
    assert _RATIO_EPS == 1e-6


@pytest.mark.parametrize(
    "columns",
    [
        {"TP2": [1.0]},  # falta TP3, Motor_current, Reservoirs
        {"TP2": [1.0], "TP3": [2.0]},  # tem o par TP2/TP3 mas não os outros
    ],
)
def test_cross_sensor_features_skipped_when_required_columns_missing(
    columns: dict[str, list[float]],
) -> None:
    pre = MetroPTPreprocessor()
    out = pre._add_cross_sensor_features(pd.DataFrame(columns))
    if "TP3" not in columns:
        assert "TP2_TP3_diff" not in out.columns
        assert "TP2_TP3_ratio" not in out.columns
    assert "work_per_pressure" not in out.columns
    assert "reservoir_drop" not in out.columns


# ---------------------------------------------------------------------------
# _add_lags — v2
# ---------------------------------------------------------------------------


def test_lags_all_nan_when_history_shorter_than_or_equal_to_lag_falls_back_to_zero() -> (
    None
):
    """
    Achado real desta task (RNF-64/RNF-65): `shift(n)` sobre um DataFrame
    com `len(df) <= n` linhas empurra TUDO pra fora da janela — a coluna
    fica inteiramente NaN, e SEM `.fillna(0.0)` depois do `.bfill()`
    (correção aplicada em `_add_lags` por esta task), `bfill()` não tem
    nenhum valor pra propagar e a coluna continuaria NaN.

    Isso não é hipotético: o `SensorBuffer` de produção
    (`feature_buffer.py`) fica "warm" com EXATAMENTE `warmup_size=15`
    linhas por padrão — o MESMO valor do `lag_long=15` padrão do
    preprocessador — e é nesse instante exato que
    `InferencePipelineService` chama `transform()` pela primeira vez. Sem
    a correção, a primeira inferência feita logo após o buffer aquecer
    receberia `TP2_lag_15`/`TP2_roc_15` como NaN, e o ONNX Runtime não
    rejeita NaN — propagaria silenciosamente.
    """
    pre = MetroPTPreprocessor(sensor_cols=["TP2"], lag_short=5, lag_long=15)
    # Exatamente 15 linhas — o boundary real do warmup_size do SensorBuffer.
    df = pd.DataFrame({"TP2": [float(i) for i in range(15)]})

    out = pre._add_lags(df)

    assert not out["TP2_lag_15"].isna().any()
    assert not out["TP2_roc_15"].isna().any()
    assert (out["TP2_lag_15"] == 0.0).all()  # fallback neutro documentado
    # roc = (TP2 - lag) / 15; com lag=0.0 (fallback), valor determinístico
    # e diferente de zero (exceto na linha TP2=0) — prova que o roc usou o
    # lag corrigido, não deixou o NaN original do shift vazar adiante.
    assert out["TP2_roc_15"].tolist() == pytest.approx(
        [(v - 0.0) / 15.0 for v in df["TP2"]]
    )


def test_lags_partial_history_still_uses_bfill_not_zero() -> None:
    """Contraste com o teste acima: quando HÁ pelo menos 1 valor válido após
    o shift (`len(df) > lag`), o comportamento continua sendo `bfill` — a
    correção desta task só afeta o caso "nenhum valor válido", nunca
    substitui o `bfill` normal por zero."""
    pre = MetroPTPreprocessor(sensor_cols=["TP2"], lag_short=2, lag_long=3)
    df = pd.DataFrame({"TP2": [10.0, 20.0, 30.0, 40.0, 50.0]})  # len=5 > lag_long=3
    out = pre._add_lags(df)

    # shift(3): [NaN, NaN, NaN, 10, 20] -> bfill -> [10, 10, 10, 10, 20]
    assert out["TP2_lag_3"].tolist() == pytest.approx([10.0, 10.0, 10.0, 10.0, 20.0])
    assert not (out["TP2_lag_3"] == 0.0).any()  # bfill, nunca o fallback 0.0


def test_lags_short_column_also_falls_back_to_zero_not_one_when_history_too_short() -> (
    None
):
    """O teste acima só exercitava `lag_long` (df tinha 15 linhas > lag_short
    5, então o `lag_short` usava `bfill` normal, nunca esse fallback).
    RNF-64: `lag_short` tem o MESMO fallback `.fillna(0.0)`, testado aqui
    isoladamente com `len(df) <= lag_short`."""
    pre = MetroPTPreprocessor(sensor_cols=["TP2"], lag_short=5, lag_long=100)
    df = pd.DataFrame({"TP2": [float(i) for i in range(5)]})  # len == lag_short
    out = pre._add_lags(df)
    assert not out["TP2_lag_5"].isna().any()
    assert (out["TP2_lag_5"] == 0.0).all()


def test_roc_fillna_replaces_nan_with_zero_not_one_when_raw_sensor_value_is_missing() -> (
    None
):
    """O `.fillna(0.0)` do próprio `roc` só tem efeito observável quando o
    valor BRUTO do sensor (não o lag) é NaN — situação real de um sensor
    que falhou momentaneamente."""
    pre = MetroPTPreprocessor(sensor_cols=["TP2"], lag_short=1, lag_long=2)
    df = pd.DataFrame({"TP2": [10.0, 20.0, np.nan, 40.0]})
    out = pre._add_lags(df)
    assert out["TP2_roc_2"].iloc[2] == 0.0


def test_lags_shift_by_configured_amount_and_backfill_the_start() -> None:
    pre = MetroPTPreprocessor(sensor_cols=["TP2"], lag_short=1, lag_long=2)
    df = pd.DataFrame({"TP2": [10.0, 20.0, 30.0, 40.0]})
    out = pre._add_lags(df)

    # shift(1): [NaN, 10, 20, 30] -> bfill -> [10, 10, 20, 30]
    assert out["TP2_lag_1"].tolist() == pytest.approx([10.0, 10.0, 20.0, 30.0])
    # shift(2): [NaN, NaN, 10, 20] -> bfill -> [10, 10, 10, 20]
    assert out["TP2_lag_2"].tolist() == pytest.approx([10.0, 10.0, 10.0, 20.0])


def test_rate_of_change_uses_the_long_lag_and_divides_by_its_size() -> None:
    pre = MetroPTPreprocessor(sensor_cols=["TP2"], lag_long=2)
    df = pd.DataFrame({"TP2": [10.0, 20.0, 30.0, 40.0]})
    out = pre._add_lags(df)

    # roc na última linha: (TP2[3] - lag_2[3]) / 2 = (40 - 20) / 2 = 10.0
    assert out["TP2_roc_2"].iloc[-1] == pytest.approx(10.0)
    # Nenhum NaN vaza mesmo nas primeiras linhas (fillna(0.0) explícito).
    assert not out["TP2_roc_2"].isna().any()


# ---------------------------------------------------------------------------
# _add_rolling_minmax — v2
# ---------------------------------------------------------------------------


def test_rolling_minmax_and_range_over_the_configured_window() -> None:
    pre = MetroPTPreprocessor(sensor_cols=["TP2"], window_minmax=3)
    df = pd.DataFrame({"TP2": [5.0, 1.0, 9.0, 2.0]})
    out = pre._add_rolling_minmax(df)

    # Janela de 3 na última linha: [1.0, 9.0, 2.0] -> min=1, max=9, range=8
    assert out["TP2_min_3"].iloc[-1] == pytest.approx(1.0)
    assert out["TP2_max_3"].iloc[-1] == pytest.approx(9.0)
    assert out["TP2_range_3"].iloc[-1] == pytest.approx(8.0)


# ---------------------------------------------------------------------------
# transform() — pipeline completo, v1 vs v2
# ---------------------------------------------------------------------------


def test_transform_v1_only_omits_v2_feature_columns() -> None:
    pre = MetroPTPreprocessor(enable_v2_features=False, sensor_cols=["TP2"])
    df = pd.DataFrame({"TP2": [1.0, 2.0, 3.0]})
    out = pre.transform(df)

    # V1: delta, std, ma_curta, ma_longa presentes.
    assert "TP2_delta" in out.columns
    assert "TP2_std_5" in out.columns
    # V2: lag/minmax/cross-sensor NUNCA aparecem quando desligado.
    assert not any(c.startswith("TP2_lag_") for c in out.columns)
    assert not any(
        c.startswith("TP2_min_") or c.startswith("TP2_max_") for c in out.columns
    )


def test_transform_v2_enabled_adds_lag_and_minmax_columns() -> None:
    pre = MetroPTPreprocessor(enable_v2_features=True, sensor_cols=["TP2"])
    df = pd.DataFrame({"TP2": [1.0, 2.0, 3.0, 4.0, 5.0]})
    out = pre.transform(df)

    assert f"TP2_lag_{pre.lag_short}" in out.columns
    assert f"TP2_lag_{pre.lag_long}" in out.columns
    assert f"TP2_min_{pre.window_minmax}" in out.columns
    assert f"TP2_max_{pre.window_minmax}" in out.columns


def test_transform_full_pipeline_never_leaves_nan_in_engineered_columns() -> None:
    """Contrato crítico para a inferência ONNX (RF-10): um NaN em QUALQUER
    feature engenheirada quebraria `predict_proba` silenciosamente (ONNX
    Runtime não valida NaN — propaga probabilidades sem sentido)."""
    pre = MetroPTPreprocessor(sensor_cols=["TP2", "TP3"])
    df = pd.DataFrame(
        {
            "TP2": [1.0, np.nan, 3.0, 4.0, 5.0, 6.0],
            "TP3": [2.0, 2.0, np.nan, 4.0, 5.0, 6.0],
        }
    )
    out = pre.transform(df)
    assert not out.isna().any().any(), out.isna().sum()[out.isna().sum() > 0]
