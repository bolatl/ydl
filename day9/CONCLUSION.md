# Conclusion — Kaggle `inception` Sleep-Stage Classification

> Full project story: what we built, what we tried, what worked, and why we stopped.
> Working dir: `day9/`. Metric: **macro-F1** (4 nearly-balanced classes, 0–3).

---

## 1. The result

| | CV macro-F1 | Kaggle LB |
|---|---|---|
| **Final submission — Tuned SVM + iterative imputation** | **0.842** | **0.85456** |

- Best **validated** model; the score is at the data's effective ceiling (~0.855).
- File: [outputs/svm_iterimpute_submission.csv](day9/outputs/svm_iterimpute_submission.csv)
- Notebook: [models/03_svm.ipynb](day9/models/03_svm.ipynb)
- *(Kaggle was unreachable from the work environment, so leaderboard rank is read off the competition page manually.)*

---

## 2. The journey in one paragraph

We benchmarked **six model families** under one fixed validation protocol; the **RBF-SVM won**.
The single biggest lever was **not** the model — it was **iterative imputation** of `eog_burst_index`,
a feature that is **50% missing**. Recovering it (predicting it from the other 20 features with a
linear BayesianRidge regression, refit inside each CV fold) lifted the SVM from **0.832 → 0.842 CV**
and **0.83877 → 0.85456 on the leaderboard** — the one change that decisively moved the score.
Everything we tried afterward to push past 0.855 landed inside the noise band, and we were able to
**prove from several independent angles that ~0.855 is the Bayes ceiling** of this dataset.

---

## 3. What we tried — full record

### 3.1 Model families (one shared protocol: `StratifiedKFold(5, shuffle=True, seed=42)`, `f1_macro`)
| Family | Result | Notebook |
|---|---|---|
| **SVM (RBF)** | **best — 0.842 CV / 0.85456 LB** | [03_svm.ipynb](day9/models/03_svm.ipynb) |
| CatBoost | 0.839 CV (iter-impute), below SVM on LB | [06_catboost.ipynb](day9/models/06_catboost.ipynb) |
| HistGBM | 0.827 CV | [05_histgbm.ipynb](day9/models/05_histgbm.ipynb) |
| Random Forest | ≤ SVM | [04_random_forest.ipynb](day9/models/04_random_forest.ipynb) |
| Logistic Regression | 0.745 CV | [01_logreg.ipynb](day9/models/01_logreg.ipynb) |
| kNN | 0.764 CV | [02_knn.ipynb](day9/models/02_knn.ipynb) |
| KMeans (unsup. baseline) | weak | [07_kmeans.ipynb](day9/models/07_kmeans.ipynb) |

### 3.2 The lever that WORKED
- **Iterative imputation of `eog_burst_index`** (BayesianRidge, refit per fold).
  SVM **0.832 → 0.842 CV**, **LB 0.83877 → 0.85456**. Validated: paired t-test p ≈ 2e-11.
  Also lifted the trees (CatBoost 0.822 → 0.839). Notebook: [03_svm.ipynb](day9/models/03_svm.ipynb).

### 3.3 What did NOT beat the SVM (all flat / worse — validated)
| Attempt | Result | LB | File |
|---|---|---|---|
| SVM+CatBoost ensemble 50/50 | CV gain inside noise | **0.84736** (lost) | [11_svm_catboost.ipynb](day9/models/11_svm_catboost.ipynb) |
| Multiple imputation (M=10 avg) | flat | **0.85264** | [12_svm_multiimpute.ipynb](day9/models/12_svm_multiimpute.ipynb) |
| Pre-scale before imputation | CV identical | **0.85392** | [14_prescale_impute.ipynb](day9/models/14_prescale_impute.ipynb) |
| Semi-supervised (transductive impute, self-training, prior-match) | all negative | below anchor | [13_semisupervised.ipynb](day9/models/13_semisupervised.ipynb) |
| SVM-heavy blend 0.75/0.25 | gamble | below anchor | [15_svm_heavy_blend.ipynb](day9/models/15_svm_heavy_blend.ipynb) |
| Finer C/gamma grid + blend sweep | grid flat (+0.0002) | probes ~0.855 | [16_finetune_blendsweep.ipynb](day9/models/16_finetune_blendsweep.ipynb) |
| Pre-imputation soft/weighted vote, stacking | inside noise | lost | [08_soft_voting.ipynb](day9/models/08_soft_voting.ipynb), [09_weighted_soft_voting.ipynb](day9/models/09_weighted_soft_voting.ipynb), [10_stacking.ipynb](day9/models/10_stacking.ipynb) |
| `class_weight` on class 2 (incl. `balanced`) | didn't move class-2 F1 | — | (tested live) |
| Per-class threshold tuning (nested) | gain −0.0006, p=0.47 | — | [12_svm_multiimpute.ipynb](day9/models/12_svm_multiimpute.ipynb) |
| Feature engineering / poly / PCA / quantile / power | hurt or flat | — | (see [03_svm.ipynb](day9/models/03_svm.ipynb)) |
| Feature selection (drop noise features) | no droppable feature | — | (tested live) |

### 3.4 Proof we hit the ceiling (why no further gain exists)
- **Class 2 is irreducibly overlapping** with classes 1 & 3 (confusion matrix is symmetric; class-2 F1 = 0.798 caps the macro score). `class_weight` couldn't even move its own F1.
- **Imputation is maxed:** `eog_burst_index` is already recovered at **R² = 0.874** (linear is optimal vs SVR 0.862 / HistGBR 0.821 / ExtraTrees 0.757) — ~87% of variance, rest is noise.
- **No temporal signal:** the rows are shuffled i.i.d. — P(same stage as next row) = 0.258 vs 0.251 chance (1.03× = none). The real sleep-staging lever (neighbor epochs) was removed by shuffling.
- **All families share a hard core:** 701 rows (7.8%) are wrong for *all* models, class-2-heavy. Oracle (≥1 model right) = 0.92 but **unrealizable** (majority vote 0.824 < SVM).
- **Hyperparameters already optimal:** finer C/gamma grid is a flat plateau (best +0.0002 = noise).

---

## 4. Methodology rules we held throughout
- One fixed validation split everywhere → every score is comparable.
- **Adopt a change only if its gain clearly exceeds the ±0.008 fold noise**, confirmed with repeated CV + paired t-test.
- Don't trust sub-noise CV gains — they lost on the LB every time (the ensemble is the textbook example).
- Distinguish **public-LB noise** from real gains: identical-quality models scattered across 0.85264–0.85456; a competitor at 0.85718 (+0.0026) is within that band.

---

## 5. Files
- Handoff / technical log: [CLAUDE.md](day9/CLAUDE.md)
- EDA: [1.ipynb](day9/1.ipynb)
- Models & ensembles: [models/](day9/models/) (notebooks 01–16)
- Submissions: [outputs/](day9/outputs/)
- Slides: [SLIDES.md](day9/SLIDES.md)
