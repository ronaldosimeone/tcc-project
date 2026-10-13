"""Matrizes de confusão dos modelos (MetroPT-3) — figuras do TCC.

1. Deriva a matriz exata dos model cards (classification_report no teste).
2. Verifica de forma independente (reexecuta ONNX/joblib no mesmo split).
3. Calcula a do Autoencoder (não existe report no card).
4. Plota.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ML = Path("C:/Users/Raphael Okuyama/Desktop/projeto-tcc/apps/ml")
OUT = Path(
    "C:/Users/Raphael Okuyama/Desktop/projeto-tcc/docs/figuras-tcc/matrizes-confusao"
)
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ML))

MODELS = {
    "Random Forest": ("model_card.json", "tabular"),
    "XGBoost": ("xgboost_v1_card.json", "tabular"),
    "MLP": ("mlp_v1_card.json", "tabular"),
    "TCN": ("tcn_v1_card.json", "janelas"),
    "BiLSTM": ("bilstm_v1_card.json", "janelas"),
    "PatchTST": ("patchtst_v1_card.json", "janelas"),
}


def cm_from_report(rep: dict) -> np.ndarray:
    """[[TN, FP], [FN, TP]] a partir de precision/recall/support por classe."""
    n0, n1 = int(rep["0"]["support"]), int(rep["1"]["support"])
    tp = round(rep["1"]["recall"] * n1)
    fn = n1 - tp
    fp = round(tp / rep["1"]["precision"] - tp) if rep["1"]["precision"] > 0 else 0
    tn = n0 - fp
    # consistência: recall da classe 0
    r0 = tn / n0
    assert abs(r0 - rep["0"]["recall"]) < 5e-4, (r0, rep["0"]["recall"])
    return np.array([[tn, fp], [fn, tp]], dtype=np.int64)


def metrics(cm: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = cm.ravel()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {
        "precision": p,
        "recall": r,
        "f1": f1,
        "specificity": tn / (tn + fp),
        "accuracy": (tn + tp) / cm.sum(),
    }


results: dict[str, dict] = {}

# ── 1. derivação a partir dos cards ──────────────────────────────────────────
for name, (card_file, kind) in MODELS.items():
    card = json.loads((ML / "models" / card_file).read_text(encoding="utf-8"))
    m = card["metrics"]
    results[name] = {
        "kind": kind,
        "threshold": card["decision_threshold"],
        "n": int(
            m["classification_report"]["0"]["support"]
            + m["classification_report"]["1"]["support"]
        ),
        "cm_default": cm_from_report(m["classification_report"]),
        "cm_tuned": cm_from_report(m["tuned_classification_report"]),
        "source": "model card (classification_report)",
    }
    print(
        name,
        "tuned",
        results[name]["cm_tuned"].tolist(),
        "default",
        results[name]["cm_default"].tolist(),
    )

# ── 2. verificação independente: modelos de janelas (ONNX) ───────────────────
import onnxruntime as ort  # noqa: E402
from src.datamodule_sequence import (  # noqa: E402
    MetroPTSequenceDataModule,
    SequenceConfig,
)
from src.datamodule_unsupervised import (  # noqa: E402
    MetroPTUnsupervisedDataModule,
    UnsupervisedConfig,
)

parquet = ML / "data" / "processed" / "metropt3.parquet"


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def run_onnx(path: Path, x: np.ndarray, batch: int | None) -> np.ndarray:
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inp = s.get_inputs()[0].name
    outs = []
    bs = batch if batch else 1024
    for i in range(0, len(x), bs):
        outs.append(s.run(None, {inp: x[i : i + bs]})[0])
    return np.concatenate(outs)


dm = MetroPTSequenceDataModule(parquet, SequenceConfig())
dm.setup()
xv = dm.val_ds.tensors[0].numpy().astype(np.float32)
yv = dm.val_ds.tensors[1].numpy()
print("val windows", xv.shape, "fault", int(yv.sum()))

onnx_files = {
    "TCN": "tcn_v1.onnx",
    "BiLSTM": "bilstm_v1.onnx",
    "PatchTST": "patchtst_v1.onnx",
}
from sklearn.metrics import confusion_matrix  # noqa: E402

for name, f in onnx_files.items():
    thr = results[name]["threshold"]
    batch = 1 if name == "PatchTST" else None  # card: fixed_batch_size = 1
    logits = run_onnx(ML / "models" / f, xv, batch)
    prob = (
        softmax(logits)[:, 1]
        if logits.shape[1] == 2
        else 1 / (1 + np.exp(-logits[:, 0]))
    )
    cm_t = confusion_matrix(yv, (prob >= thr).astype(int), labels=[0, 1])
    cm_d = confusion_matrix(yv, (prob >= 0.5).astype(int), labels=[0, 1])
    d = results[name]
    d["verify_tuned"] = cm_t
    d["verify_default"] = cm_d
    print(name, "recomputed tuned", cm_t.tolist(), "vs card", d["cm_tuned"].tolist())
    print(
        name, "recomputed default", cm_d.tolist(), "vs card", d["cm_default"].tolist()
    )

# ── 3. Autoencoder ───────────────────────────────────────────────────────────
ae_card = json.loads(
    (ML / "models" / "autoencoder_v1_card.json").read_text(encoding="utf-8")
)
dm_ae = MetroPTUnsupervisedDataModule(parquet, UnsupervisedConfig())
dm_ae.setup()
xa = dm_ae.val_ds.tensors[0].numpy().astype(np.float32)
ya = dm_ae.val_ds.tensors[1].numpy()
rec = run_onnx(ML / "models" / "autoencoder_v1.onnx", xa, None)
mse = ((rec - xa) ** 2).mean(axis=(1, 2))
thr_ae = ae_card["mse_threshold"]
cm_ae = confusion_matrix(ya, (mse >= thr_ae).astype(int), labels=[0, 1])
from sklearn.metrics import roc_auc_score  # noqa: E402

auc_ae = roc_auc_score(ya, mse)
print("Autoencoder", cm_ae.tolist(), "AUC", round(auc_ae, 4), "thr", thr_ae)
results["Autoencoder"] = {
    "kind": "janelas",
    "threshold": thr_ae,
    "n": int(len(ya)),
    "cm_tuned": cm_ae,
    "cm_default": cm_ae,
    "auc": float(auc_ae),
    "source": "recalculado (ONNX no conjunto de validação)",
    "ae": True,
}

# ── 4. verificação XGBoost (tabular, joblib) ─────────────────────────────────
try:
    import joblib
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from src.preprocessing import MetroPTPreprocessor

    card = json.loads(
        (ML / "models" / "xgboost_v1_card.json").read_text(encoding="utf-8")
    )
    names = card["feature_names"]
    raw = pd.read_parquet(parquet)
    eng = MetroPTPreprocessor(enable_v2_features=False).transform(raw)
    X = eng[names].astype(np.float32)
    y = raw["anomaly"].astype(int)
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    mdl = joblib.load(ML / "models" / "xgboost_v1.joblib")
    est = (
        mdl.named_steps[list(mdl.named_steps)[-1]]
        if hasattr(mdl, "named_steps")
        else mdl
    )
    proba = est.predict_proba(X_te)[:, 1]
    thr = results["XGBoost"]["threshold"]
    results["XGBoost"]["verify_tuned"] = confusion_matrix(
        y_te, (proba >= thr).astype(int), labels=[0, 1]
    )
    results["XGBoost"]["verify_default"] = confusion_matrix(
        y_te, (proba >= 0.5).astype(int), labels=[0, 1]
    )
    print(
        "XGBoost recomputed tuned",
        results["XGBoost"]["verify_tuned"].tolist(),
        "vs card",
        results["XGBoost"]["cm_tuned"].tolist(),
    )
except Exception as e:  # noqa: BLE001
    print("XGBoost verify skipped:", type(e).__name__, e)


# ── 5. persistir ─────────────────────────────────────────────────────────────
def j(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError


(OUT / "matrizes_confusao.json").write_text(
    json.dumps(results, default=j, ensure_ascii=False, indent=1), encoding="utf-8"
)
print("saved json")
