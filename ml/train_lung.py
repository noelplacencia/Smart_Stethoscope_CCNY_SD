"""
train_lung.py
--------
Trains a Random Forest classifier on the extracted ICBHI features.
Evaluates with cross-validation and saves the trained model.

Output:
    ml/data/rf_model_lung.joblib   — trained model (copy this to the RPi)
    ml/data/confusion_matrix_lung.png
    ml/data/roc_curve_lung.png
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
from sklearn.preprocessing import label_binarize, StandardScaler
from imblearn.over_sampling import SMOTE

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_CSV = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/features_lung.csv"
MODEL_OUT    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/rf_model_lung.joblib"
SCALER_OUT   = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/scaler_lung.joblib"
CM_OUT       = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/confusion_matrix_lung.png"
ROC_OUT      = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/roc_curve_lung.png"

# ── Label names ────────────────────────────────────────────────────────────────
LABEL_NAMES = ["normal", "crackle", "wheeze", "both"]


def load_data():
    df = pd.read_csv(FEATURES_CSV)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    print(f"Loaded {len(df)} windows, {X.shape[1]} features each")
    print(f"Class distribution:")
    for i, name in enumerate(LABEL_NAMES):
        count = np.sum(y == i)
        pct   = count / len(y) * 100
        print(f"  {name:<10} {count:>5}  ({pct:.1f}%)")
    return X, y


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,       # 200 trees
        max_depth=None,         # grow until pure
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",    # sqrt of features per split
        class_weight="balanced",# penalize minority class errors more
        random_state=42,
        n_jobs=-1,              # use all CPU cores
    )
    model.fit(X_train, y_train)
    return model


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    print("\n── Classification Report ─────────────────────────────────────")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))

    # Overall accuracy
    acc = np.mean(y_pred == y_test)
    print(f"Overall accuracy : {acc:.1%}")

    # ROC-AUC (one-vs-rest)
    y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    auc   = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    print(f"Macro ROC-AUC   : {auc:.3f}")

    return y_pred, y_prob


def plot_confusion_matrix(y_test, y_pred):
    cm      = confusion_matrix(y_test, y_pred)
    cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")

    ax.set_xticks(range(len(LABEL_NAMES)))
    ax.set_yticks(range(len(LABEL_NAMES)))
    ax.set_xticklabels(LABEL_NAMES, fontsize=11)
    ax.set_yticklabels(LABEL_NAMES, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Smart Stethoscope — Lung Sound Confusion Matrix\n"
                 "(count / % of true class)", fontsize=12, pad=12)

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "navy"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=10, color=color)

    # Legend: true class totals
    totals = cm.sum(axis=1)
    legend_labels = [f"{name}: {n} samples" for name, n in zip(LABEL_NAMES, totals)]
    handles = [plt.Rectangle((0, 0), 1, 1, fc="none", ec="none") for _ in legend_labels]
    ax.legend(handles, legend_labels, title="True class totals",
              loc="upper right", bbox_to_anchor=(1.35, 1), fontsize=9, title_fontsize=9)

    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nConfusion matrix saved to: {CM_OUT}")


def plot_roc_curves(model, X_test, y_test):
    y_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    y_prob = model.predict_proba(X_test)

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#1D9E75", "#534AB7", "#D85A30", "#E5A020"]

    for i, (name, color) in enumerate(zip(LABEL_NAMES, colors)):
        RocCurveDisplay.from_predictions(
            y_bin[:, i], y_prob[:, i],
            name=name, color=color, ax=ax
        )

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_title("Smart Stethoscope — ROC Curves (one-vs-rest)", fontsize=13, pad=12)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_OUT, dpi=150)
    plt.close()
    print(f"ROC curves saved to     : {ROC_OUT}")


def cross_validate(model, X, y):
    print("\n── 5-Fold Cross Validation ───────────────────────────────────")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # StratifiedKFold preserves class proportions in each fold
    scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    print(f"F1 macro per fold : {[f'{s:.3f}' for s in scores]}")
    print(f"Mean F1 macro     : {scores.mean():.3f} ± {scores.std():.3f}")


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
    print("  Smart Stethoscope — Random Forest Training")
    print("=" * 60 + "\n")

    # Load
    X, y = load_data()

    # Split — 80% train, 20% test, stratified to preserve class ratios
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set : {len(X_train)} windows")
    print(f"Test set  : {len(X_test)} windows")

    # Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # SMOTE — balance minority classes on training set only
    print("\nApplying SMOTE to training set...")
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    print(f"Training set after SMOTE: {len(X_train)} windows")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:<10} {np.sum(y_train == i)}")

    # Train
    print("\nTraining Random Forest...")
    model = train_model(X_train, y_train)
    print("Done.")

    # Evaluate on held-out test set
    y_pred, y_prob = evaluate(model, X_test, y_test)

    # Cross-validation on full dataset
    cross_validate(model, X, y)

    # Feature importance
    print_feature_importance(model)

    # Plots
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curves(model, X_test, y_test)

    # Save model and scaler
    joblib.dump(model, MODEL_OUT)
    joblib.dump(scaler, SCALER_OUT)
    print(f"\nModel saved to : {MODEL_OUT}")
    print(f"Scaler saved to: {SCALER_OUT}")
    print("\nCopy both .joblib files to the Raspberry Pi when ready.")


if __name__ == "__main__":
    main()