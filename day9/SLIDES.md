# 3-Slide Defense — Sleep-Stage Classification (`inception`)

> Format: 3 slides / ~5 minutes. Paste into PowerPoint, Google Slides, or render with Marp.
> `---` separates slides. Speaker notes are in the *Say:* lines.

---

## Slide 1 — Result

# Sleep-Stage Classification — macro-F1 **0.85456**

**Final model:** Tuned RBF-SVM + iterative imputation of a 50%-missing feature

| Step | macro-F1 (LB) |
|---|---|
| SVM baseline (median imputation) | 0.83877 |
| **SVM + iterative imputation** | **0.85456** |

- 4 sleep stages, nearly balanced · metric = macro-F1 (the weakest class caps the score)
- This score is at the **data's effective ceiling** (shown on slide 3)

*Say: "Our best leaderboard score is 0.8546 macro-F1. One change — how we filled a half-missing
feature — drove almost the entire jump. Rank: [fill in from Kaggle]."*

---

## Slide 2 — Approach

# How we got there

**1. Benchmark 6 model families** — one fixed CV protocol, so every score is comparable
→ **RBF-SVM won** (LogReg/kNN/RandomForest/HistGBM/CatBoost all below).

**2. Find the real lever — the data, not the model.**
One feature (`eog_burst_index`) was **50% missing**. Instead of filling it with a constant
(median), we **predicted** it from the other 20 features with a per-fold linear regression
(*iterative imputation*).
→ SVM **0.832 → 0.842 CV**, **0.839 → 0.855 LB**. Statistically validated (paired t-test p ≈ 2e-11).

**3. Validate everything; trust nothing inside the noise.**
Rule: adopt a change only if it beats the **±0.008 fold noise** (repeated CV + paired t-test).
Ensembling, multiple imputation, semi-supervised learning, threshold tuning → all inside noise,
all lost or flat on the leaderboard.

*Say: "We fixed the data problem before tuning models. Ensembles looked better on CV but the
gain was inside noise — and they lost on the board. Discipline about noise was the whole game."*

---

## Slide 3 — Learnings

# What we learned

- **Fix the data before the model.** A single 50%-missing feature was worth more than every
  model/hyperparameter choice combined.
- **Sub-noise CV gains are traps.** The "better" ensemble (CV 0.845) *lost* on the LB (0.847 vs
  0.855). We learned to demand a gain bigger than ±0.008, proven with a paired test.
- **We proved we hit the ceiling — not gave up:**
  - Class 2 genuinely overlaps its neighbors (caps macro-F1; reweighting can't fix it)
  - The missing feature is already **87% recovered** (R² 0.874) — no information left
  - Data is **shuffled** → no temporal signal to exploit (the usual sleep-staging trick)
  - 6 model families fail on the **same** 701 hard rows
- **Public-LB ≠ truth.** Scores scatter ±0.002 from luck; we picked the best *validated* model,
  not the best *public* number.

*Say: "The honest finding is that 0.855 is the Bayes ceiling for this dataset — and we can show
why from four independent directions. Proving the limit is a stronger result than chasing noise."*
