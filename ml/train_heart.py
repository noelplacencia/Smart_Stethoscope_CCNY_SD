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

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_CSV = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/features_heart.csv"
MODEL_OUT    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/rf_model_heart.joblib"
CM_OUT       = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/confusion_matrix_heart.png"
ROC_OUT      = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/roc_curve_heart.png"

LABEL_NAMES = ["absent", "present"]


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
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("\n── Classification Report ─────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    acc = np.mean(y_pred == y_test)
    auc = roc_auc_score(y_test, y_prob)
    print(f"Overall accuracy : {acc:.1%}")
    print(f"ROC-AUC          : {auc:.3f}")

    return y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Smart Stethoscope — Heart Sound Confusion Matrix", fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=150)
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
