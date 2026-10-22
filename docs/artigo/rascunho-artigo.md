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

## 2. Related work **[partially filled — full comparison in `docs/related_work_comparison.md`; PRISMA citations still TODO]**

**Validation on MetroPT.** Published results split into two groups. Studies that split readings at random (80/20 or undeclared) report accuracy/F1 of 0.95–1.00 for Random Forest, XGBoost, LSTM and ensembles (IJACSA 2025; Najjar et al., 2023; several public repositories). Neighbouring 10 s readings are near-duplicates and the test asks whether a reading lies *inside* an already-seen failure, so these scores reflect leakage, which we reproduce in §4.1 (0.999 → 0.59). Studies with chronological or event-held-out protocols agree with ours. An open 34-configuration benchmark finds that ML and deep detectors match but do not beat CUSUM/EWMA charts (best: 4/4 failures, 0.71 false alarms/week), that unreported leak episodes in training blind autoencoders, and that "4/4 is compatible with a true rate as low as 40 %". A purged TCN/XGBoost study with July held out detects nothing at the default threshold and concludes the task is unsolved. On MetroPT2, Ribeiro et al. (2025) detect both failures before the LPS alarm but show the Flowmeter alone separates them and recommend dropping it, as we do. None of these evaluates transfer to another unit or live training–serving parity. We run CUSUM/EWMA under the same nested protocol and gate as the learned models (`docs/statistical_baselines.md`: EWMA 4/6 failures, lowest cost, about 2× the false alarms of BiLSTM, no comparison significant) and replace training-set cleaning with a pre-registered label audit used only as a secondary analysis (`docs/label_audit.md`).

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

### 4.5b Diagnosis of the real failures (`docs/rag_diagnosis_benchmark.md`)

Operator-written questions overstate what the assistant can do: in production it receives only what the platform measures. For each of the nine documented failures (MetroPT-3 ×4, MetroPT 2022 ×3, MetroPT2 ×2) we took the first 30 min of the failure, ran the online sensor-occlusion RCA of the served model, and asked the assistant through the production path. Retrieval relevance is defined by WAP-5 task code per component family (provisional, pending expert validation). Sensor evidence alone identifies the failed component no better than chance: balanced identification 0.28–0.33 against 0.33 for a uniform guess, and **none of the three non-air-leak failures (air-dryer valve, two oil leaks) is identified by any evidence-only query or embedder**; the 0.67 raw accuracy of the occlusion query equals the majority-class rate, because every query lands on the generic main-compressor chunk. The cause is structural, not a tuning issue: the RCA blames the same shared sensors (oil temperature, motor current, dryer discharge pressure) for every failure type, and the corpus is a periodic-maintenance schedule with no fault-finding table. With the production embedder and τ = 0.6 the grounding gate abstains on all 36 evidence/oracle queries and no citation is fabricated; with bge-m3 the operator-style query clears the gate, and the 3 B model then names the wrong component in 56 % of plans and the right one in 11 %. Detection belongs to the ML layer; failure-type diagnosis is not supported by this corpus and is not claimed.

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

### 4.7b Every model under one nested protocol (`docs/model_evaluation_leakfree.md`)

The legacy artefacts — and the Optuna hyper-parameters of XGBoost — had been selected on a random split. Re-running all eight models (RF, XGBoost, logistic regression, MLP, TCN, BiLSTM, PatchTST, Conv1D autoencoder) under one nested protocol — outer purged walk-forward (6 out-of-time folds), inner leave-one-failure-out for hyper-parameters, Platt calibration and the RN-02 threshold, 3 seeds — RF drops from 3 to 2 of 6 failures once its leaked hyper-parameters are removed. In this round XGBoost had the lowest expected cost (3.3 of 6, 46.5), but the paired one-sided Wilcoxon over folds gave p = 0.125 over RF — and the round-6 audit (§4.7d) later changed both the features and the false-alarm metric, after which BiLSTM leads and no candidate passes the gate. Static split conformal covers only 12–54 % of failure rows out of time; class-conditional adaptive conformal inference restores 88–91 % coverage for every model with mean set sizes of 1.03–1.26. Remaining-useful-life regression stays at chance (C-index 0.48 vs 0.65 for a calendar baseline) even with seven failures.

### 4.7c Failures no model has seen (`docs/metropt2_unseen_failures.md`)

MetroPT2 (Zenodo 7766691) contains two failures — an air leak (4 June 2022) and an oil leak (11–14 July 2022) — that start after the end of every training set. On the 35 days it shares with MetroPT 2022 its TP3 pressure is uncorrelated with ours (r ≈ 0), so it is a different train's compressor: the test is out-of-sample in time *and* in asset, and the served artefacts were scored unchanged (ONNX, card thresholds, persistence k = 6). All seven served models detect the air leak, before the train's own low-pressure (LPS) alarm for six of them, the tree models with no false-alarm episode. Only the four sequence/neural models (MLP, TCN, BiLSTM, PatchTST) and the physics twin — recalibrated on the asset's own healthy weeks, without labels — detect the oil leak, about 57 h before LPS fires; the deployed random forest, XGBoost and the autoencoder miss it. An apparent two-hour early warning by MLP and TCN on the air leak turned out to be an isolated 1–5 min alarm followed by 2.5 h of silence, which motivated reporting the onset of the first *sustained* episode (≥ 10 min) alongside the first alarm. Retraining the same nested recipe only on data before the single 2022 oil leak isolates what that one example buys: **without it, none of the seven models detects the unseen oil leak; with it, the four sequence/neural models raise a sustained alarm about 55 h before LPS**, on a different asset. Tree models and the autoencoder miss it either way, while the air leak is detected by all seven in both variants. With two failures, these are counts, not rates.

**Offline ≠ live, until fixed.** Replaying the real failures through the exact serving path (simulator → feature buffer → preprocessor → ONNX) showed the models detecting most scenario × model pairs, yet an operator saw none: the dashboard and push alerts used fixed probability bands (0.35/0.65/0.70) for every model, whereas the leak-free decision thresholds range from 0.08 (MLP) to 0.905 (XGBoost), so detections at P = 0.2–0.5 were painted NORMAL. Two further training/serving skews surfaced: sequence models were scored on left-padded windows before 60 real rows existed (a constant segment never seen in training, which produced a false alarm 19 s after every start), and switching scenarios spliced two recordings in one window. Risk is now expressed relative to the active model's calibrated threshold, windowed models are scored only on full real windows, and a replay starts from the asset's own preceding history; after the fixes and the final round-6 artefacts the serving path detects 30 of 42 pairs (`apps/backend/benchmarks/live_replay_matrix.json`; PatchTST 6/6, MLP and BiLSTM 5/6, random forest, XGBoost and TCN 4/6, autoencoder 2/6), and live, BiLSTM turned the unseen MetroPT2 air leak CRITICAL at replay step 311, against 307 predicted by the harness.

### 4.7d An external methodological review, audited (round 6)

An independent review of the training methodology was checked point by point against the code and data. Its two most serious concerns did not apply — Platt calibration is fitted on out-of-fold predictions of every held-out row (real prevalence), and the threshold never sees the test fold — but the audit found four real defects, all fixed before re-running the entire protocol: (i) rolling windows counted *rows*, so they crossed 331 time gaps > 60 s in 2020 (160 > 1 h); features are now segmented at gaps, with a byte-identical serving mirror; (ii) a backward fill in imputation (future values; inactive since the data have no nulls) was made causal; (iii) class weight and negative subsampling were fixed constants and are now part of the nested search; (iv) **the event metric had a loophole**: an always-on alarm was a single episode touching the failure, "detected with zero false alarms", and the cost-driven threshold search of the autoencoder converged to it (46 % of test time in alarm). False alarms are now counted on alarm time outside the failure windows (one per started hour, i.e. re-notification). Under the corrected protocol BiLSTM has the lowest expected cost (3.0/6 failures, 57.6), followed by PatchTST (3.0/6, 61.0) and XGBoost (2.7/6, 61.7); the autoencoder falls to 1.0/6 (117.1); no candidate beats the random-forest champion under the RN-05 gate (best p = 0.19). Which model is best depends on the cost ratio (10:1 → logistic regression/MLP, 100:1 → BiLSTM/XGBoost, 1000:1 → autoencoder/PatchTST).

The audit also produced three studies. A per-event analysis shows that the highest pooled correlate, DV_pressure, is driven by three 2020 air leaks (Cohen's d 8.3, 3.0, 9.4; ≈ 0 for the other four failures); its ablation helps XGBoost but *improves* the random forest when removed, and alone it detects only 2020 leaks. An anticipation target ("a failure starts within H", H = 30 min–2 h, failure-in-progress rows excluded) answers the detection-versus-prediction question with a chance control: models trained for it **never** switch an alarm on inside the pre-failure window, and their apparent hits are explained by being in alarm 17–83 % of the time (random windows hit 9–91 %). With these sensors and seven failures there is no learnable 30 min–2 h precursor; the platform is a detector of failures in progress, with minutes of lead.

### 4.8 Energy (`models/eval_energy.json`, RNF-83)

CodeCarbon (offline, Brazilian grid ≈ 98 gCO₂/kWh), CPU and GPU energy measured by hardware counters, idle power subtracted. Training a model costs 0.02–0.15 Wh; ONNX inference ≈ 0.09 mWh per 1,000 predictions (≈ 0.3 Wh per asset-year at 0.1 Hz). One grounded repair plan from the local 3 B LLM costs 0.072 Wh (0.007 gCO₂) — **as much as ~800,000 predictions**. The footprint of the system is the advisory layer, not the detector.

## 5. Threats to validity

See `docs/pesquisa/questoes-e-hipoteses.md` §Ameaças: one compressor family (two periods, and the 2022 recording maps two sensors to different signals under the same names), 3 test folds on MetroPT-3 alone (no comparison can reach p < 0.05; the pooled 6-fold protocol lifts this floor to p = 0.016), draft RAG annotations pending expert validation, proxy edge measurements.

## 6. Conclusion **[TODO]**

## Artifact availability

Code, configuration, result JSONs, benchmark cases and the reproduction script: **[Zenodo DOI after release]**. MetroPT-3 is CC BY 4.0; third-party manuals are not redistributed.
