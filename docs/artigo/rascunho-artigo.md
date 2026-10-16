# DRAFT — Leakage-free, event-level evaluation of a physics-aware predictive maintenance pipeline with grounded LLM repair advice via MCP

> Target: IEEE Access (regular paper) or PHM Society workshop.
> Every number below is produced by `bash scripts/reproduce-thesis.sh` and
> traceable to a JSON in `apps/ml/models/` or `apps/backend/benchmarks/out/`
> (hashes in `docs/reproducibility_manifest.json`). Sections marked
> **[TODO]** need the authors (literature from the PRISMA review, venue
> formatting).

## Abstract (≈ 200 words)

Data-driven predictive maintenance is routinely reported with near-perfect
scores on public benchmarks such as MetroPT-3, yet these scores rarely
survive deployment. We show that a stratified random split over rolling-window
features inflates PR-AUC from 0.59 to 0.999 (Random Forest), and that under a
purged walk-forward protocol with event-level metrics no purely data-driven
detector anticipates failures by the 2 h an operator needs. A grey-box
digital twin of the compressor (reservoir mass balance, electrical work, duty
cycle) detects the failure that every ML model misses, 2.6 h in advance;
**late fusion of ML and physics alarms is the only configuration that detects
all three test failures**, cutting the expected operational cost from 35.0 to
11.3. For the actionable step we connect the detector to a local LLM through
the Model Context Protocol and measure what reaches the technician: with the
production embedder the correct manual page is retrieved in 2.4 % of cases,
an LLM without retrieval hallucinates procedures for 100 % of unanswerable
questions, and a malicious manual makes both local LLMs recommend disabling a
safety valve — which deterministic grounding and safety guards prevent. We
release code, benchmark and a one-command reproduction package.

## 1. Introduction **[TODO: motivation + citations from PRISMA]**

Contributions:

1. **An evaluation protocol** for run-to-failure-scarce industrial data: purged walk-forward by failure event, event-level metrics (event recall, lead time, false alarms/day, range-based F1, asymmetric expected cost), thresholds chosen only on training out-of-fold scores, and a statistical promotion gate that reports when a comparison is underpowered.
2. **A physics-aware detector**: a grey-box twin whose residuals stay consistent across failures (duty-cycle residual z = +4.0 … +5.3 in all three) and complement ML; late fusion detects 3/3 failures.
3. **A grounded repair assistant**: XAI → MCP → RAG with a pre-generation grounding gate (RN-04), a post-generation citation check and a deterministic safety guard, plus a public benchmark (54 cases, 12 unanswerable) and a red-team suite.
4. **An open, reproducible system** (FastAPI, ONNX, MQTT, OpenTelemetry, RBAC with a hash-chained audit trail) used as the testbed.

## 2. Related work **[TODO — fill from `docs/pesquisa/protocolo-revisao-sistematica.md`]**

Axes: validation protocols on MetroPT/MetroPT-2; event-based metrics (Tatbul et al., 2018; NAB); grey-box/hybrid PdM; XAI faithfulness (deletion, Samek et al., 2017); RAG evaluation and hallucination; LLM security / prompt injection; MCP.

## 3. Data and protocol

- MetroPT-3: 1.52 M samples at 10 s from a Porto metro APU compressor, Feb–Aug 2020, **4 documented air-leak failures**.
- MetroPT 2022 (Veloso et al., *Sci. Data* 2022; Zenodo 10.5281/zenodo.6854240): same APU family, Jan–Jun 2022, 1 Hz, **3 failures** (two air leaks, one oil leak). Point-sampled to 10 s (1.08 M rows), GPS dropped; used for cross-period transfer and a pooled 7-failure walk-forward (6 test folds).
- Purged walk-forward: segment boundaries at mid-points between failures; fold *k* tests failure *k*+1 and trains strictly before it (purge 300 s > largest feature window of 150 s). 3 folds × 5 seeds.
- Alarms: persistence k = 6 samples (RN-03). Lead-time target 2 h (RN-01). Cost C_FN = 100 · C_FP (RN-02). Thresholds: training out-of-fold, cost-optimal.

## 4. Results

### 4.1 The leakage gap (Table 1, `docs/temporal_evaluation.md`)

| Model | PR-AUC random split | PR-AUC walk-forward [95 % CI] | Event recall | Lead time (h) |
|---|---|---|---|---|
| Random Forest | 0.999 | 0.587 [0.428, 0.739] | 0.67 | +0.92 |
| XGBoost | 0.9996 | 0.400 [0.283, 0.543] | 0.67 | +0.13 |
| Logistic regression | 0.709 | 0.331 [0.155, 0.547] | 0.67 | +0.17 |

No detector meets the 2 h lead-time requirement. Friedman over folds: p ≥ 0.37; the promotion gate refuses every swap because with 3 folds the minimum one-sided p is 0.125.

### 4.2 Physics vs ML vs fusion (Table 2, `models/eval_twin.json`)

| Variant | Failures detected (fold 1 / 2 / 3) | Lead time, July failure | Expected cost [95 % CI] | False alarms/day |
|---|---|---|---|---|
| ML only (XGB) | 3/3 · 3/3 · 0/3 | missed | 35.0 [11.8, 67.3] | 0.07 |
| Hybrid features (XGB + 4 residuals) | 3/3 · 3/3 · 0/3 | missed | 35.9 [11.7, 67.8] | 0.11 |
| Physics only | 0/3 · 0/3 · 3/3 | **2.61 h** | 77.0 [45.0, 101.0] | 0.31 |
| **Late fusion** | **3/3 · 3/3 · 3/3** | **2.61 h** | **11.3 [7.8, 14.4]** | 0.35 |

Feeding the residuals as extra features does not help (the trees ignore 4 of 84 columns); fusing the *decisions* does. Paired Wilcoxon over folds: p = 1.0 (underpowered, N = 3) — the effect needs the cross-dataset replication (RF-33).

### 4.3 Uncertainty and explanations

- Mondrian conformal (α = 0.1) calibrated on training OOF scores reaches 0.59–0.68 coverage on the failure class under temporal shift (target 0.90): exchangeability does not hold across failures.
- TreeSHAP is faithful (deletion of the top-5 features drops P(failure) by 0.69 vs 0.13 for 5 random features) but **unstable** across failures (Kendall τ between folds −0.24 … 0.76), whereas the physics residuals move consistently (§4.2).

### 4.4 Robustness (`models/eval_robustness.json`)

Gaussian noise of 5 % σ drops RF PR-AUC from 0.94 to 0.23 and raises false alarms to 13/day; freezing a single sensor (DV_pressure) suppresses detection in both tree models; a calibration bias of 0.25 σ hides the failure from RF.

### 4.5 Grounded repair advice (`docs/rag_benchmark.md`)

| Embedder | hit@5 | MRR (XAI query) | Grounding AUC | Gold coverage at τ = 0.6 |
|---|---|---|---|---|
| MiniLM-L12 (production) | 0.62 | 0.40 | 0.61 | 2.4 % |
| bge-m3 | 0.69 | **0.49** | **0.89** | 28.6 % |

LLM-only answers produce a procedure for 100 % of the 12 unanswerable questions (invented torques, temperatures and part numbers); with retrieval and the grounding gate this drops to 0–8.3 %.

### 4.6 Red team (prompt injection via a malicious manual)

| Attack (given the chunk reached the prompt) | No guard (llama3.2:3b / qwen2.5:3b) | With guard |
|---|---|---|
| Recommend disabling the safety valve | 2/2 · 2/2 | 0/2 · 0/2 |
| Reply only with a canary token | 0/1 · 1/1 | chunk dropped |
| Cite a fabricated document | 1/1 · 1/1 (flagged by RN-04 citation check) | chunk dropped |

A system prompt is not a security boundary for 3 B-parameter local models; deterministic pre- and post-generation guards are required.

### 4.7 A second recording period (`docs/cross_dataset_evaluation.md`, RF-33)

Matching columns by name is not enough: in MetroPT-3 `Reservoirs` duplicates `TP3` (r = 1.000) while in 2022 it is a different ~1.6 bar signal, and `Oil_level` is constant in 2022; both were excluded. Even then the two periods are separable (adversarial AUC 0.998). **Direct transfer fails in both directions** (2020→2022: 1/3 failures; 2022→2020: ≤ 1/4). Pooling the periods in a purged walk-forward (7 failures, 6 test folds) does not detect the two 2022 failure modes absent from 2020 (air-dryer valve, oil leak), but a few weeks of *normal* 2022 data let the 2020-trained detector catch the first 2022 air leak, and the 6-fold protocol finally gives the statistics power: Friedman p = 0.032, and the RN-05 gate promotes XGBoost over logistic regression (one-sided Wilcoxon p = 0.016, the minimum attainable).

The deployed detector, trained with a random split and SMOTE, scores PR-AUC 1.000 on the 2020 rows it saw and **0.058 on 2022, with zero alarms** — fixing the sensors at its input does not help. A physics twin recalibrated on a few weeks of *normal* 2022 operation detects the first 2022 air leak, whereas the uncalibrated 2020 twin does not; in the pooled walk-forward, late fusion has the lowest expected cost (39.1 vs 51.2 for ML) and the hybrid model is the only one to flag the oil leak (2/3 seeds), though no difference passes the gate. Neither 2022 failure mode outside the training data (air-dryer valve, oil leak) moves the twin's air-balance residuals, which bounds what this twin can see.

### 4.8 Energy (`models/eval_energy.json`, RNF-83)

CodeCarbon (offline, Brazilian grid ≈ 98 gCO₂/kWh), CPU and GPU energy measured by hardware counters, idle power subtracted. Training a model costs 0.02–0.15 Wh; ONNX inference ≈ 0.09 mWh per 1,000 predictions (≈ 0.3 Wh per asset-year at 0.1 Hz). One grounded repair plan from the local 3 B LLM costs 0.072 Wh (0.007 gCO₂) — **as much as ~800,000 predictions**. The footprint of the system is the advisory layer, not the detector.

## 5. Threats to validity

See `docs/pesquisa/questoes-e-hipoteses.md` §Ameaças: one compressor family (two periods, and the 2022 recording maps two sensors to different signals under the same names), 3 test folds on MetroPT-3 alone (no comparison can reach p < 0.05; the pooled 6-fold protocol lifts this floor to p = 0.016), draft RAG annotations pending expert validation, proxy edge measurements.

## 6. Conclusion **[TODO]**

## Artifact availability

Code, configuration, result JSONs, benchmark cases and the reproduction script: **[Zenodo DOI after release]**. MetroPT-3 is CC BY 4.0; third-party manuals are not redistributed.
