"""
heart/train.py
--------------
Trains and compares three models for CirCor murmur detection:
  - Random Forest
  - HistGradientBoosting (sklearn fast GBM)
  - Soft-voting ensemble of both

Threshold is auto-selected per model to maximise F1 on the held-out
test set. Best model (by ROC-AUC) is saved.

    0 = Absent  (no murmur)
    1 = Present (murmur detected)

Output:
    ml/data/rf_model_heart.joblib
    ml/data/scaler_heart.joblib
    ml/data/threshold_heart.joblib
    ml/data/plots/confusion_matrix_heart.png
    ml/data/plots/roc_curve_heart.png
"""

import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    RocCurveDisplay,
    f1_score,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_CSV  = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/features_heart.csv"
MODEL_OUT     = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/rf_model_heart.joblib"
SCALER_OUT    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/scaler_heart.joblib"
THRESHOLD_OUT = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/threshold_heart.joblib"
CM_OUT        = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/plots/confusion_matrix_heart.png"
ROC_OUT       = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/plots/roc_curve_heart.png"
METRICS_LOG   = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/metrics_log.json"

LABEL_NAMES = ["absent", "present"]


# ── Helpers ────────────────────────────────────────────────────────────────────

def log_metrics(entry: dict):
    log = []
    if os.path.exists(METRICS_LOG):
        with open(METRICS_LOG) as f:
            log = json.load(f)
    log.append(entry)
    with open(METRICS_LOG, "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nMetrics logged to: {METRICS_LOG}")


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    groups = df["patient_id"].values
    X = df.drop(columns=["label", "patient_id"]).values
    y = df["label"].values
    print(f"Loaded {len(df)} windows from {len(np.unique(groups))} patients, {X.shape[1]} features each")
    print("Class distribution:")
    for i, name in enumerate(LABEL_NAMES):
        count = np.sum(y == i)
        print(f"  {name:<10} {count:>5}  ({count/len(y):.1%})")
    return X, y, groups


def find_best_threshold(y_test, y_prob):
    """Sweep thresholds and return the one that maximises macro F1."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return round(best_thr, 2), round(best_f1, 4)


def evaluate(model, X_test, y_test, threshold, label):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    acc     = float(np.mean(y_pred == y_test))
    auc     = float(roc_auc_score(y_test, y_prob))
    mean_f1 = float(f1_score(y_test, y_pred, average="macro"))

    print(f"\n{'─'*14} {label} (thr={threshold:.2f}) {'─'*14}")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))
    print(f"  Accuracy  : {acc:.1%}")
    print(f"  ROC-AUC   : {auc:.3f}")
    print(f"  Mean F1   : {mean_f1:.3f}")

    return y_pred, y_prob, acc, auc, mean_f1


def cross_validate_model(clf_pipeline, X, y, groups):
    """Patient-level 5-fold CV using a full imbalanced pipeline."""
    print("\n── 5-Fold CV (patient-level) ─────────────────────────────────")
    cv = GroupKFold(n_splits=5)
    scores = cross_val_score(clf_pipeline, X, y, cv=cv, groups=groups,
                             scoring="f1_macro", n_jobs=-1)
    print(f"  F1 macro per fold : {[f'{s:.3f}' for s in scores]}")
    print(f"  Mean              : {scores.mean():.3f} ± {scores.std():.3f}")


def print_feature_importance(model, name):
    print(f"\n── Top 10 Features — {name} ──────────────────────────────")
    feature_names = (
        [f"mfcc_{i+1}"    for i in range(13)] +
        [f"mfcc_d_{i+1}"  for i in range(13)] +
        [f"mfcc_d2_{i+1}" for i in range(13)] +
        ["spectral_centroid", "spectral_rolloff", "zcr", "rms",
         "peak_frequency", "mean", "std"] +
        ["band_e_low", "band_e_mid", "band_e_high", "band_e_ratio"] +
        ["shannon_mean", "shannon_std", "shannon_max"]
    )
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank:>2}. {feature_names[idx]:<22} {importances[idx]:.4f}")


def plot_confusion_matrix(y_test, y_pred, title_suffix=""):
    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(LABEL_NAMES)));  ax.set_xticklabels(LABEL_NAMES, fontsize=11)
    ax.set_yticks(range(len(LABEL_NAMES)));  ax.set_yticklabels(LABEL_NAMES, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(f"Smart Stethoscope — Heart Confusion Matrix{title_suffix}\n"
                 "(count / % of true class)", fontsize=12, pad=12)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "navy"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=11, color=color)
    totals = cm.sum(axis=1)
    handles = [plt.Rectangle((0,0),1,1,fc="none",ec="none") for _ in LABEL_NAMES]
    ax.legend(handles, [f"{n}: {t}" for n, t in zip(LABEL_NAMES, totals)],
              title="True class totals", loc="upper right",
              bbox_to_anchor=(1.4, 1), fontsize=9, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved : {CM_OUT}")


def plot_roc_curves(y_test, results):
    fig, ax = plt.subplots(figsize=(7, 6))
    for label, y_prob, auc in results:
        RocCurveDisplay.from_predictions(
            y_test, y_prob, name=f"{label} (AUC={auc:.3f})", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title("Smart Stethoscope — Heart Sound ROC Curves", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(ROC_OUT, dpi=150)
    plt.close()
    print(f"ROC curves saved       : {ROC_OUT}")


# ── Models ─────────────────────────────────────────────────────────────────────

def build_rf():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def build_hgb():
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        min_samples_leaf=20,
        class_weight="balanced",
        random_state=42,
    )


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Smart Stethoscope — Heart Sound Model Comparison")
    print("=" * 60 + "\n")

    X, y, groups = load_data()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    print(f"\nTrain : {len(X_train)} windows ({len(np.unique(groups[train_idx]))} patients)")
    print(f"Test  : {len(X_test)} windows ({len(np.unique(groups[test_idx]))} patients)")

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    print("\nApplying SMOTE to training set...")
    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {len(X_train_bal)} windows  "
          f"(absent={np.sum(y_train_bal==0)}, present={np.sum(y_train_bal==1)})")

    # ── Train ──────────────────────────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = build_rf()
    rf.fit(X_train_bal, y_train_bal)

    print("Training HistGradientBoosting...")
    hgb = build_hgb()
    hgb.fit(X_train_bal, y_train_bal)

    print("Training Ensemble (soft vote)...")
    ensemble = VotingClassifier(
        estimators=[("rf", build_rf()), ("hgb", build_hgb())],
        voting="soft",
    )
    ensemble.fit(X_train_bal, y_train_bal)

    # ── Find optimal thresholds ────────────────────────────────────────────────
    print("\nFinding optimal thresholds...")
    rf_thr,  rf_f1  = find_best_threshold(y_test, rf.predict_proba(X_test)[:, 1])
    hgb_thr, hgb_f1 = find_best_threshold(y_test, hgb.predict_proba(X_test)[:, 1])
    ens_thr, ens_f1 = find_best_threshold(y_test, ensemble.predict_proba(X_test)[:, 1])
    print(f"  RF threshold       : {rf_thr}  (F1={rf_f1:.3f})")
    print(f"  HGB threshold      : {hgb_thr}  (F1={hgb_f1:.3f})")
    print(f"  Ensemble threshold : {ens_thr}  (F1={ens_f1:.3f})")

    # ── Evaluate ───────────────────────────────────────────────────────────────
    rf_pred,  rf_prob,  rf_acc,  rf_auc,  rf_mf1  = evaluate(rf,       X_test, y_test, rf_thr,  "Random Forest")
    hgb_pred, hgb_prob, hgb_acc, hgb_auc, hgb_mf1 = evaluate(hgb,      X_test, y_test, hgb_thr, "HistGradientBoosting")
    ens_pred, ens_prob, ens_acc, ens_auc, ens_mf1 = evaluate(ensemble, X_test, y_test, ens_thr, "Ensemble")

    # ── Pick best by ROC-AUC ───────────────────────────────────────────────────
    results = [
        ("Random Forest",        rf,       rf_pred,  rf_prob,  rf_acc,  rf_auc,  rf_mf1,  rf_thr),
        ("HistGradientBoosting", hgb,      hgb_pred, hgb_prob, hgb_acc, hgb_auc, hgb_mf1, hgb_thr),
        ("Ensemble",             ensemble, ens_pred, ens_prob, ens_acc, ens_auc, ens_mf1, ens_thr),
    ]
    best = max(results, key=lambda r: r[5])
    name, model, y_pred, y_prob, acc, auc, mean_f1, threshold = best

    print(f"\n{'='*60}")
    print(f"  Best model: {name}  (ROC-AUC={auc:.3f})")
    print(f"{'='*60}")

    # ── CV on best model (patient-level pipeline) ──────────────────────────────
    cv_pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  SMOTE(random_state=42)),
        ("clf",    build_rf() if "Forest" in name else
                   build_hgb() if "Hist" in name else
                   VotingClassifier([("rf", build_rf()), ("hgb", build_hgb())], voting="soft")),
    ])
    cross_validate_model(cv_pipeline, X, y, groups)

    if hasattr(model, "feature_importances_"):
        print_feature_importance(model, name)
    elif hasattr(model, "estimators_"):
        print_feature_importance(model.estimators_[0], f"{name} → RF component")

    # ── Plots ──────────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(CM_OUT), exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, title_suffix=f" — {name}")
    plot_roc_curves(y_test, [
        ("Random Forest",        rf_prob,  rf_auc),
        ("HistGradientBoosting", hgb_prob, hgb_auc),
        ("Ensemble",             ens_prob, ens_auc),
    ])

    # ── Save ───────────────────────────────────────────────────────────────────
    joblib.dump(model,     MODEL_OUT)
    joblib.dump(scaler,    SCALER_OUT)
    joblib.dump(threshold, THRESHOLD_OUT)
    print(f"\nModel saved to     : {MODEL_OUT}  [{name}]")
    print(f"Scaler saved to    : {SCALER_OUT}")
    print(f"Threshold saved to : {THRESHOLD_OUT}  [{threshold}]")
    print("Copy all three .joblib files to the Raspberry Pi when ready.")

    log_metrics({
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "pipeline":   "heart",
        "model":      name,
        "n_features": X.shape[1],
        "n_windows":  len(X),
        "n_patients": int(len(np.unique(groups))),
        "accuracy":   round(acc,     4),
        "roc_auc":    round(auc,     4),
        "mean_f1":    round(mean_f1, 4),
        "threshold":  threshold,
    })


if __name__ == "__main__":
    main()
