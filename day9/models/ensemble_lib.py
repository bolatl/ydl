"""
Shared helpers for the ensemble notebooks (08, 09, 10).

Why this file exists
--------------------
All three ensemble notebooks need the SAME ingredient: the out-of-fold (OOF)
probabilities of each base model on the training set, plus each base model's
probabilities on the test set. Computing those is the only slow step. We compute
them ONCE here and cache them to ``outputs/preds/`` as .npy files, so every
notebook just loads arrays from disk and runs in seconds.

OOF (out-of-fold) probabilities
-------------------------------
For each training row we want a prediction made by a model that did NOT see that
row during training. ``cross_val_predict`` gives exactly this: it runs the same
5-fold split, and each row's probability comes from the fold where it was held
out. These OOF probabilities are an honest, leak-free basis for (a) measuring
ensemble quality, (b) searching voting weights, and (c) training a stacking
meta-model.

Test probabilities are produced by refitting each model on ALL training rows.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score
from catboost import CatBoostClassifier

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..")
PRED_DIR = os.path.join(HERE, "..", "outputs", "preds")
os.makedirs(PRED_DIR, exist_ok=True)

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
FEATURES = [c for c in train.columns if c not in ("id", "sleep_stage")]
y = train["sleep_stage"].to_numpy()
CLASSES = np.array([0, 1, 2, 3])

# The one validation split reused everywhere, so every score is comparable.
cv = StratifiedKFold(5, shuffle=True, random_state=42)


def builders():
    """Factory dict: name -> function returning a fresh, TUNED base estimator.

    Params come straight from the tuning we did in notebooks 03/05/06.
    SVM uses probability=True so it can output calibrated class probabilities
    (needed for soft voting / stacking); this makes it slower to train.
    Tree models (catboost, histgbm) take raw features incl. NaN; the
    distance/linear models (svm, logreg, knn) get imputed + scaled inside a pipeline.
    """
    return {
        "svm": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            # CalibratedClassifierCV turns SVM's margins into calibrated probabilities
            # (the recommended replacement for the deprecated SVC(probability=True)).
            CalibratedClassifierCV(
                SVC(kernel="rbf", C=10, gamma=0.01, random_state=42), ensemble=False),
        ),
        "catboost": lambda: CatBoostClassifier(
            loss_function="MultiClass", iterations=600, depth=7, learning_rate=0.04,
            l2_leaf_reg=3, random_state=42, verbose=0, thread_count=-1,
        ),
        "histgbm": lambda: HistGradientBoostingClassifier(
            learning_rate=0.1, max_iter=600, max_leaf_nodes=63, l2_regularization=1.0,
            random_state=42, early_stopping=True, validation_fraction=0.15, n_iter_no_change=30,
        ),
        "logreg": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(max_iter=2000),
        ),
        "knn": lambda: make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            KNeighborsClassifier(n_neighbors=15),
        ),
    }


def get_proba(name, force=False, verbose=True):
    """Return (oof_proba, test_proba), each shape (n_rows, 4), with disk caching.

    oof_proba[i]  = class probabilities for train row i, predicted by a model
                    trained on the other folds (leak-free).
    test_proba[j] = class probabilities for test row j, from a model trained on
                    ALL training rows.
    """
    oof_path = os.path.join(PRED_DIR, f"{name}_oof.npy")
    test_path = os.path.join(PRED_DIR, f"{name}_test.npy")
    if not force and os.path.exists(oof_path) and os.path.exists(test_path):
        if verbose:
            print(f"[{name}] loaded cached probabilities")
        return np.load(oof_path), np.load(test_path)

    build = builders()[name]
    if verbose:
        print(f"[{name}] computing OOF + test probabilities (this is the slow part)...")
    oof = np.asarray(cross_val_predict(build(), train[FEATURES], y, cv=cv,
                                       method="predict_proba", n_jobs=1))
    model = build()
    model.fit(train[FEATURES], y)
    tp = np.asarray(model.predict_proba(test[FEATURES]))
    np.save(oof_path, oof)
    np.save(test_path, tp)
    if verbose:
        print(f"[{name}] done. OOF macro-F1 = {f1_score(y, oof.argmax(1), average='macro'):.4f}")
    return oof, tp


def macro_f1(proba, weights=None):
    """Macro-F1 of argmax predictions from a probability matrix.

    ``weights`` (length-4) multiplies each class column before argmax; this is
    how we do per-class threshold tuning for the macro metric.
    """
    p = proba if weights is None else proba * np.asarray(weights)
    return f1_score(y, p.argmax(1), average="macro")


def tune_class_weights(oof_proba, grid=None, passes=3):
    """Deterministic coordinate-ascent search for per-class probability weights
    that maximize OOF macro-F1. Boosting a class's weight makes the model predict
    it more often -- useful for the weak class 2. Returns (weights, best_oof_f1).

    NOTE: scaling all weights equally doesn't change argmax, so only relative
    weights matter. We search one class at a time, repeating ``passes`` times.
    """
    if grid is None:
        grid = np.round(np.linspace(0.5, 1.8, 27), 3)
    w = np.ones(oof_proba.shape[1])
    best = macro_f1(oof_proba, w)
    for _ in range(passes):
        for c in range(len(w)):
            best_factor = w[c]
            for f in grid:
                cand = w.copy(); cand[c] = f
                s = macro_f1(oof_proba, cand)
                if s > best:
                    best, best_factor = s, f
            w[c] = best_factor
    return w, best
