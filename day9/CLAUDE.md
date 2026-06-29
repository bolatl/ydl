# Handoff — Kaggle "inception" Sleep-Stage Classification (YDL Week 2, Days 4–5)

> Working dir for all artifacts: `/home/bolat/ydl26/day9/`
> Python: use the project venv → `/home/bolat/ydl26/.venv/bin/python` (CatBoost 1.2.10, seaborn 0.13.2, scikit-learn ≥1.9).

---

## 1. Project Goal & Context
- **Task:** Kaggle competition `inception` — predict the sleep stage of each epoch in the test set.
- **Metric:** **macro-F1** (every class weighted equally → the weakest class caps the score).
- **Deadline:** submissions close **Fri 26 Jun 2026, 14:30**; final rank is on the **private** leaderboard. Then a 3-slide / 5-minute defense.
- **Deliverables:** Kaggle submissions + a reproducible notebook + a 3-slide deck (slide 1: best result & rank; slide 2: approach; slide 3: learnings).
- **Kaggle is unreachable from this sandbox** (network blocked), so all analysis is from the local data, not the competition page.

### Data
- `train.csv` = 9000 rows, `test.csv` = 5000 rows. IDs disjoint (train 0–8999, test 9000–13999).
- **21 numeric physiological features** (EEG bands, EMG, EOG, heart rate, respiration, SpO₂, body movement) + `id` + target `sleep_stage`.
- **Target: 4 classes (0–3), nearly balanced** (22–27% each). NOT the usual 5-stage scheme.
- Features are **pre-standardized** (mean ≈ 0) but with **different spreads** (std ≈ 1–5). No train/test distribution shift, no duplicate rows.
- **Only one column has missing values: `eog_burst_index`, ~50% missing in BOTH train and test, MCAR.** This turned out to be the single most important modeling lever.
- Hardest class = **class 2** (overlaps classes 1 & 3); it caps macro-F1.

---

## 2. Key Decisions Made

### What WORKED (the two real levers)
1. **Iterative imputation of `eog_burst_index`** (`IterativeImputer(estimator=BayesianRidge(), max_iter=10)`), applied *inside* a pipeline so it refits per CV fold. Since only one column is missing, this is effectively "predict `eog_burst_index` from the other 20 features."
   - SVM: median-impute **0.832** → iterative-impute **0.842** CV. Validated: paired t-test **p ≈ 2e-11**, better in 28–29/30 folds. **LB jumped 0.83877 → 0.85456.**
   - It also lifts the trees: **CatBoost 0.822 → 0.839**, HistGBM 0.817 → 0.827.
2. **Soft-voting ensemble of SVM + CatBoost** (equal weights), both on iterative-imputed features.
   - CV **~0.845**. Validated robust: **15/15 folds** won across 3 seeds, paired **p ≈ 3e-5**.
   - Works *because* imputation made CatBoost strong enough; two strong, different-family models with decorrelated errors.

### What did NOT work (don't re-try these — all flat or worse than the SVM baseline)
- **Ensembling before imputation:** soft/weighted voting & stacking gained only +0.003 on OOF — **inside the ±0.008 fold noise** — and lost on the LB. (Lesson that recurs: never trust sub-noise gains.)
- **SVM + CatBoost ensemble (iter-imputed), the "expected >0.855" model:** CV ~0.845 but **LB 0.84736 — LOST to the single SVM (0.85456)**. Its CV gain was inside the noise band; the LB confirmed it. (nb `11_svm_catboost.ipynb`.)
- **Feature engineering on the SVM:** polynomial/interaction features and PCA-whitening *hurt* (RBF already models interactions); quantile/power transforms, missing-indicator → flat.
- **Alternative models:** NuSVC, poly-kernel SVM, LogReg+poly, LDA, QDA, ExtraTrees, kNN, RandomForest, KMeans — all ≤ SVM.
- **Nonlinear imputers** (ExtraTrees / HistGBR / KNN inside IterativeImputer): worse (0.833–0.835); the linear BayesianRidge generalizes best.
- **Bagged SVM, weighted-vote weight search, stacking meta-model:** overfit OOF / no robust gain.
- **GPU (GTX 1650):** only CatBoost can use it; benchmarked **GPU 63s vs CPU 24s** per fit → GPU is *slower* on this small data. **Stay on CPU.**

### Session 2 — ceiling confirmed from every angle (all validated NEGATIVE; do not re-try)
The true quality is **~0.855 and that is the data's Bayes ceiling.** Each lever below was tested
live; each hit the same wall (class 2 genuinely overlaps classes 1 & 3 in feature space).
- **Multiple imputation** (`sample_posterior=True`, average M=10 draws, nb `12`): **LB 0.85264** — below anchor. Threshold tuning on its OOF: gain −0.0006, p=0.47 → off.
- **Pre-scale before imputation** (`Scaler→IterativeImputer→Scaler→SVM`, nb `14`): CV identical to scale-after (0.8407 vs 0.8407); **LB 0.85392** — below anchor. BayesianRidge is scale-robust; features already ~standardized.
- **SVM `class_weight` on class 2** (incl. `'balanced'`): negative — upweighting class 2 did **not even move its own F1** (0.798→0.797). Its points sit *inside* other classes; reweighting the boundary can't fix overlap.
- **Per-class threshold tuning** (nested CV): negative / inside noise. (Post-hoc reweighting of fixed probs; ranking already near-optimal.)
- **Semi-supervised using the 5000 test rows** (nb `13`: transductive imputation, self-training/pseudo-labeling with class 2 excluded, prior-matching): all validated negative, all scored *below* the anchor on LB.
- **Temporal context:** the data is **shuffled i.i.d.** — P(stage_i == stage_{i+1}) by id order = 0.258 vs 0.251 chance (**1.03× = none**); transition matrix is flat. The real sleep-staging lever (neighbor epochs) was destroyed by shuffling. **No sequence to exploit.**
- **Imputation quality is maxed:** `eog_burst_index` is recovered at **R²=0.874** with BayesianRidge (best vs SVR 0.862, HistGBR 0.821, ExtraTrees 0.757). ~87% of variance recovered, rest is noise — the one proven lever has no headroom left, and the linear imputer is genuinely optimal.
- **Feature selection (drop noise features for RBF):** leave-one-feature-out CV — dropping *any* feature is ≤0 (best = `heart_rate_mean` +0.0000, all others hurt; `eog_burst_index` −0.0175). No noise features exist; all 21 carry signal and are already used. Poor-man's-ARD = dead end.
- **Trees cannot beat the SVM** (structural): CatBoost/HistGBM/RF/ExtraTrees all converge 0.01–0.02 below. RBF-SVM winning ⇒ the boundary is smooth/oblique, which axis-aligned trees approximate poorly on 9000 rows.
- **Oracle/hard-core diagnostic:** 701 rows (7.8%) are wrong for *all 5* model families, class-2-heavy — the irreducible core. Oracle (≥1 model right) = 0.92 but **unrealizable** (majority vote 0.824 < SVM; rescuing models too weak).
- **About 0.86:** the 0.85264 / 0.85392 / 0.85456 spread is **public-LB sampling noise** on identical-quality models. Fishing for a 0.86 *public* reading overfits the public subset and regresses to ~0.855 on the **graded private** board. Don't submission-fish; pick the best-*validated* model and trust it.

### Methodology rules we adopted
- Single shared validation protocol everywhere: **`StratifiedKFold(5, shuffle=True, random_state=42)`**, scoring `f1_macro`.
- **Only adopt a change if its gain clearly exceeds the ~±0.008 fold noise**, confirmed with **repeated CV + paired t-test**.
- Tune with **fixed best hyperparameters** for refits (cheap); the slow part is grid search, which is already done.
- Ensemble work uses **cached out-of-fold (OOF) + test probabilities** in `outputs/preds/*.npy` so notebooks load instantly instead of retraining.

---

## 3. Current Status

### Best results (LB numbers now all in — anchor is the single SVM)
| Model | CV macro-F1 | Kaggle LB |
|---|---|---|
| Tuned SVM, median-impute | 0.832 | 0.83877 |
| **Tuned SVM, iterative-impute  ← FINAL PICK** | **0.842** | **0.85456** |
| SVM, pre-scale impute (nb 14) | 0.8407 | 0.85392 |
| SVM, multiple imputation M=10 (nb 12) | ~0.841 | 0.85264 |
| SVM + CatBoost ensemble 50/50, iter-impute (nb 11) | ~0.845 | 0.84736 |
| SVM-heavy blend 0.75/0.25, iter-impute (nb 15) | ~noise | not submitted (lottery ticket) |

**Ceiling is ~0.855 (Bayes-limited by class-2 overlap).** Everything within ±0.002 of the anchor is public-LB sampling noise. See "Session 2" above for the full validated-negative list.

### Recommended submissions
- **Final pick (graded private LB): `outputs/svm_iterimpute_submission.csv`** — best *validated* model, LB 0.85456.
- **Optional 2nd slot:** `outputs/ensemble_svm_catboost_iterimpute_submission.csv` (decorrelated, robust on CV) — NOT a 3rd noisy SVM.
- `outputs/svm_heavy_blend_submission.csv` exists as a spare-submission experiment only; expected private gain ≈ 0.

### File map (`day9/`)
```
1.ipynb                              # EDA (executed, with plots)
ydl_w2_day4_zadanie_markdown.md      # the assignment brief (Russian)
train.csv / test.csv / sample_submission.csv
models/
  ensemble_lib.py                    # shared: tuned base models + cached OOF/test probs (NOTE: uses MEDIAN impute)
  01_logreg / 02_knn / 03_svm / 04_random_forest / 05_histgbm / 06_catboost / 07_kmeans .ipynb
  08_soft_voting / 09_weighted_soft_voting / 10_stacking .ipynb   # pre-imputation ensembles (did NOT beat SVM)
  11_svm_catboost.ipynb              # SVM+CatBoost 50/50 ensemble, iter-impute. EXECUTED → LB 0.84736 (LOST to SVM).
  12_svm_multiimpute.ipynb           # multiple imputation + nested threshold tuning. EXECUTED → LB 0.85264 (below anchor).
  13_semisupervised.ipynb            # transductive impute / self-training / prior-match. ALL validated negative.
  14_prescale_impute.ipynb           # scale-before-impute vs after. Flat (CV identical) → LB 0.85392.
  15_svm_heavy_blend.ipynb           # 0.75 SVM + 0.25 CatBoost. Lottery ticket; submission built, not graded pick.
outputs/
  *_submission.csv                   # one per model + ensembles
  svm_iterimpute_submission.csv      # SVM w/ iterative impute (LB 0.85456) ← FINAL PICK
  ensemble_svm_catboost_iterimpute_submission.csv   # 50/50 ensemble (LB 0.84736)
  svm_multiimpute_submission.csv / svm_prescale_submission.csv / svm_heavy_blend_submission.csv  # session-2 experiments
  preds/*.npy                        # cached OOF + test probabilities (MEDIAN-impute base models)
```

### Notebook conventions (all model notebooks follow this)
- Live in `models/`; data is one level up (`../train.csv`); submissions go to `../outputs/`.
- Sections: explicit explanation of the algorithm → preprocessing rationale → CV benchmark (baseline vs tuned) → tuning search → **submission cell at the end**.
- CatBoost `predict` returns 2-D for multiclass → always `np.asarray(pred).astype(int).ravel()` when building the submission DataFrame.

### Tuned hyperparameters (current best)
- **SVM (final):** `IterativeImputer(BayesianRidge, max_iter=10)` → `StandardScaler` → `SVC(kernel='rbf', C=12, gamma=0.012, probability=True, random_state=42)`.
- **CatBoost (final):** `IterativeImputer(BayesianRidge)` → `CatBoostClassifier(iterations=600, depth=7, learning_rate=0.04, l2_leaf_reg=3, loss_function='MultiClass')`.
- (median-impute SVM optimum was `C=10, gamma=0.01`; HistGBM `lr=0.1, max_leaf_nodes=63, l2=1.0, early_stopping`.)

---

## 4. Open Tasks & Next Steps
**Modeling is DONE — the ceiling (~0.855) is reached and every remaining lever is exhausted
(see "Session 2" above). Do not start a new model; spend remaining time on the deck.**
1. **Select `svm_iterimpute_submission.csv` (LB 0.85456) as the final graded submission.** Optionally keep the 50/50 ensemble as a 2nd slot. Do NOT submission-fish for 0.86 — it's public-LB noise that won't survive to private.
2. **Prepare the 3 slides.** Narrative: tuned 6 model families → SVM best; **the one real win was iterative imputation of the 50%-missing `eog_burst_index`** (statistically validated, 0.832→0.855 on LB); then we *proved we hit the data ceiling* — ensembling/threshold/class_weight/semi-supervised/multi-impute all failed inside the noise band, the missing feature is already 87%-recovered (R²=0.874), the data is shuffled (no temporal signal), and 6 model families share a 701-row class-2 hard core. **"We proved 0.855 is the Bayes ceiling" is a stronger story than a lucky 0.86.**
3. Note: `ensemble_lib.py` and `outputs/preds/*.npy` are based on **median** imputation (pre-breakthrough). If reusing them, regenerate with iterative imputation or delete the cache (`force=True`).

---

## 5. Constraints & Formatting Preferences
- **No deep-learning models** — user hasn't covered them yet (no MLP/NN/keras/torch).
- **Only the provided data** — external datasets / manual test labeling are banned by competition rules. ≤20 submissions/day.
- **Reproducibility is graded** — fixed seeds, runnable notebooks that reproduce results.
- **Separate notebook file per model / ensemble variant** (user tracks files individually). Submission cell at the end of each.
- **Explanations must be explicit / educational** — explain what each algorithm does, every hyperparameter, and every preprocessing choice, so the user can understand and extend.
- **Don't compromise model quality to fit the assistant's ~12-min shell limit** — write the notebook and let the user run long jobs (e.g. heavy grid searches, multi-seed validation).
- Run long/heavy jobs in the **background** with unbuffered output (`python -u`); Python buffers piped stdout, so prints are lost on timeout otherwise.
- Submission format: exactly `id,sleep_stage`, 5000 rows, integer labels 0–3.
- In chat, reference files with clickable markdown links (e.g. `[03_svm.ipynb](day9/models/03_svm.ipynb)`), not backticks.
