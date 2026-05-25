"""
train_heart.py
--------------
Trains a Random Forest on the CirCor heart sound features (binary classification).

    0 = Absent  (no murmur)
    1 = Present (murmur detected)

Output:
    ml/data/rf_model_heart.joblib   ← copy to RPi
    ml/data/confusion_matrix_heart.png
    ml/data/roc_curve_heart.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_CSV = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/features_heart.csv"
MODEL_OUT    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/rf_model_heart.joblib"
CM_OUT       = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/confusion_matrix_heart.png"
ROC_OUT      = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/roc_curve_heart.png"

LABEL_NAMES = ["absent", "present"]
THRESHOLD   = 0.3    # lower than default 0.5 — prioritises catching murmurs over false positives


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    print(f"Loaded {len(df)} windows, {X.shape[1]} features each")
    print("Class distribution:")
    for i, name in enumerate(LABEL_NAMES):
        count = np.sum(y == i)
        pct   = count / len(y) * 100
        print(f"  {name:<10} {count:>5}  ({pct:.1f}%)")
    return X, y


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    print("\n── Classification Report ─────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    acc = np.mean(y_pred == y_test)
    auc = roc_auc_score(y_test, y_prob)
    print(f"Decision threshold : {THRESHOLD}")
    print(f"Overall accuracy   : {acc:.1%}")
    print(f"ROC-AUC            : {auc:.3f}")

    return y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred):
    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES, fontsize=11)
    ax.set_yticklabels(LABEL_NAMES, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Smart Stethoscope — Heart Sound Confusion Matrix\n"
                 "(count / % of true class)", fontsize=12, pad=12)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "navy"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=11, color=color)

    totals = cm.sum(axis=1)
    legend_labels = [f"{name}: {n} samples" for name, n in zip(LABEL_NAMES, totals)]
    handles = [plt.Rectangle((0, 0), 1, 1, fc="none", ec="none") for _ in legend_labels]
    ax.legend(handles, legend_labels, title="True class totals",
              loc="upper right", bbox_to_anchor=(1.4, 1), fontsize=9, title_fontsize=9)

    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix saved to: {CM_OUT}")


def plot_roc_curve(y_test, y_prob):
    fig, ax = plt.subplots(figsize=(7, 6))
    RocCurveDisplay.from_predictions(y_test, y_prob, name="murmur (present)", ax=ax)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title("Smart Stethoscope — Heart Sound ROC Curve", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(ROC_OUT, dpi=150)
    plt.close()
    print(f"ROC curve saved to       : {ROC_OUT}")


def cross_validate(model, X, y):
    print("\n── 5-Fold Cross Validation ───────────────────────────────────")
    cv     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
    print(f"F1 per fold : {[f'{s:.3f}' for s in scores]}")
    print(f"Mean F1     : {scores.mean():.3f} ± {scores.std():.3f}")


def print_feature_importance(model):
    print("\n── Top 10 Most Important Features ───────────────────────────")
    feature_names = [f"mfcc_{i+1}" for i in range(13)] + \
                    ["spectral_centroid", "spectral_rolloff", "zcr", "rms"]
    importances = model.feature_importances_
    indices     = np.argsort(importances)[::-1][:10]
    for rank, idx in enumerate(indices, 1):
        print(f"  {rank:>2}. {feature_names[idx]:<22} {importances[idx]:.4f}")


def main():
    print("=" * 60)
    print("  Smart Stethoscope — Heart Sound RF Training")
    print("=" * 60 + "\n")

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain : {len(X_train)} windows")
    print(f"Test  : {len(X_test)} windows")

    # Scale
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # SMOTE — balance classes on training set only
    print("\nApplying SMOTE to training set...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"Training set after SMOTE: {len(X_train)} windows")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:<10} {np.sum(y_train == i)}")

    print("\nTraining Random Forest...")
    model = train_model(X_train, y_train)
    print("Done.")

    y_pred, y_prob = evaluate(model, X_test, y_test)
    cross_validate(model, X, y)
    print_feature_importance(model)
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)

    joblib.dump(model, MODEL_OUT)
    print(f"\nModel saved to: {MODEL_OUT}")
    print("Copy rf_model_heart.joblib to the Raspberry Pi when ready.")


if __name__ == "__main__":
    main()
