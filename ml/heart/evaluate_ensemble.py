"""
ml/heart/evaluate_ensemble.py
------------------------------
Patient-level soft ensemble: RF+HGB + MobileNetV2 CNN.

For each patient in the test set:
  - RF side : mean of all 3-second window probabilities
  - CNN side: mean of all 5-second window probabilities
  - Ensemble: 0.5 * RF_prob + 0.5 * CNN_prob

Uses the same GroupShuffleSplit (random_state=42) as both training scripts
to ensure test patients are identical to what each model saw.

Output:
    ml/data/plots/confusion_matrix_heart_ensemble.png
    metrics appended to ml/data/metrics_log.json
"""

import json
import os
from collections import defaultdict
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

# ── Paths ──────────────────────────────────────────────────────────────────────
FEATURES_CSV = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/features_heart.csv"
RF_MODEL     = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/rf_model_heart.joblib"
RF_SCALER    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/scaler_heart.joblib"
CNN_MODEL    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/cnn_model_heart.pth"
MEL_CACHE    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/mel_cache_heart_5s.npz"
CM_OUT       = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/plots/confusion_matrix_heart_ensemble.png"
METRICS_LOG  = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/metrics_log.json"

IMG_SIZE    = 224
BATCH_SIZE  = 64
LABEL_NAMES = ["absent", "present"]


# ── Dataset ────────────────────────────────────────────────────────────────────

class MelDataset(Dataset):
    def __init__(self, mels, labels):
        self.mels   = mels
        self.labels = labels

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, i):
        mel = self.mels[i].copy()
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)
        t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )
        return t.squeeze(0).repeat(3, 1, 1), int(self.labels[i])


# ── Helpers ────────────────────────────────────────────────────────────────────

def find_best_threshold(y_true, y_prob):
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return round(best_thr, 2)


def plot_confusion_matrix(y_test, y_pred, title_suffix=""):
    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(LABEL_NAMES))); ax.set_xticklabels(LABEL_NAMES, fontsize=11)
    ax.set_yticks(range(len(LABEL_NAMES))); ax.set_yticklabels(LABEL_NAMES, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(f"Smart Stethoscope — Heart Ensemble{title_suffix}\n"
                 "(count / % of true class)", fontsize=12, pad=12)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "navy"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=11, color=color)
    totals = cm.sum(axis=1)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="none", ec="none") for _ in LABEL_NAMES]
    ax.legend(handles, [f"{n}: {t}" for n, t in zip(LABEL_NAMES, totals)],
              title="True class totals", loc="upper right",
              bbox_to_anchor=(1.4, 1), fontsize=9, title_fontsize=9)
    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved: {CM_OUT}")


def log_metrics(entry):
    log = []
    if os.path.exists(METRICS_LOG):
        with open(METRICS_LOG) as f:
            try:
                log = json.load(f)
            except json.JSONDecodeError:
                log = []
    log.append(entry)
    with open(METRICS_LOG, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Metrics logged to: {METRICS_LOG}")


def evaluate_patient_level(label, y_patient, y_prob):
    thr    = find_best_threshold(y_patient, y_prob)
    y_pred = (y_prob >= thr).astype(int)
    auc    = float(roc_auc_score(y_patient, y_prob))
    acc    = float(np.mean(y_pred == y_patient))
    mf1    = float(f1_score(y_patient, y_pred, average="macro", zero_division=0))
    print(f"\n{'─'*14} {label} (thr={thr:.2f}, patient-level) {'─'*14}")
    print(classification_report(y_patient, y_pred, target_names=LABEL_NAMES, zero_division=0))
    print(f"  Accuracy : {acc:.1%}")
    print(f"  ROC-AUC  : {auc:.3f}")
    print(f"  Mean F1  : {mf1:.3f}")
    return y_pred, y_prob, acc, auc, mf1, thr


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Smart Stethoscope — Heart Ensemble Evaluation")
    print("  (patient-level: RF + CNN soft vote)")
    print("=" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── RF side ───────────────────────────────────────────────────────────────
    print("\nLoading RF features ...")
    df      = pd.read_csv(FEATURES_CSV)
    groups  = df["patient_id"].values
    X       = df.drop(columns=["label", "patient_id"]).values
    y       = df["label"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx   = next(gss.split(X, y, groups))
    test_patients = set(groups[test_idx])
    print(f"  Test set: {len(test_idx)} windows, {len(test_patients)} patients")

    scaler       = joblib.load(RF_SCALER)
    rf           = joblib.load(RF_MODEL)
    X_test_rf    = scaler.transform(X[test_idx])
    rf_probs_win = rf.predict_proba(X_test_rf)[:, 1]
    rf_pids      = groups[test_idx]
    rf_labels    = y[test_idx]

    patient_rf    = defaultdict(list)
    patient_label = {}
    for prob, pid, lbl in zip(rf_probs_win, rf_pids, rf_labels):
        patient_rf[pid].append(prob)
        patient_label[pid] = int(lbl)

    # ── CNN side ──────────────────────────────────────────────────────────────
    print("\nLoading CNN mel cache ...")
    cache      = np.load(MEL_CACHE)
    mels       = cache["mels"]
    mel_labels = cache["labels"]
    mel_pids   = cache["patient_ids"]

    test_mask  = np.isin(mel_pids, list(test_patients))
    mels_test  = mels[test_mask]
    pids_test  = mel_pids[test_mask]
    print(f"  CNN test windows: {test_mask.sum()} (same {len(test_patients)} patients)")

    ckpt  = torch.load(CNN_MODEL, map_location=device)
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()

    loader = DataLoader(
        MelDataset(mels_test, mel_labels[test_mask]),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
    )

    cnn_probs_win = []
    with torch.no_grad():
        for inputs, _ in loader:
            outputs = model(inputs.to(device))
            probs   = torch.softmax(outputs, dim=1).cpu().numpy()
            cnn_probs_win.extend(probs[:, 1])
    cnn_probs_win = np.array(cnn_probs_win)

    patient_cnn = defaultdict(list)
    for prob, pid in zip(cnn_probs_win, pids_test):
        patient_cnn[pid].append(prob)

    # ── Patient-level aggregation ─────────────────────────────────────────────
    # Some short recordings produce no 5s CNN windows; drop those patients
    patient_ids = sorted(set(patient_rf.keys()) & set(patient_cnn.keys()))
    dropped = len(patient_label) - len(patient_ids)
    if dropped:
        print(f"  ({dropped} patients dropped — no CNN windows at 5s)")
    y_patient   = np.array([patient_label[p] for p in patient_ids])
    rf_pat      = np.array([np.mean(patient_rf[p])  for p in patient_ids])
    cnn_pat     = np.array([np.mean(patient_cnn[p]) for p in patient_ids])
    ens_pat     = 0.5 * rf_pat + 0.5 * cnn_pat

    print(f"\nPatients evaluated: {len(patient_ids)}  "
          f"(absent={int((y_patient==0).sum())}, present={int((y_patient==1).sum())})")

    # ── Evaluate each ─────────────────────────────────────────────────────────
    _, _, rf_acc,  rf_auc,  rf_mf1,  _ = evaluate_patient_level("RF+HGB",    y_patient, rf_pat)
    _, _, cnn_acc, cnn_auc, cnn_mf1, _ = evaluate_patient_level("CNN",       y_patient, cnn_pat)
    y_pred, _, ens_acc, ens_auc, ens_mf1, ens_thr = evaluate_patient_level(
        "RF+HGB + CNN Ensemble", y_patient, ens_pat
    )

    print(f"\n{'='*60}")
    print(f"  Summary (patient-level)")
    print(f"{'='*60}")
    print(f"  {'Model':<22} {'Acc':>7}  {'AUC':>6}  {'F1':>6}")
    print(f"  {'─'*22}  {'─'*6}  {'─'*5}  {'─'*5}")
    print(f"  {'RF+HGB':<22} {rf_acc:>7.1%}  {rf_auc:>6.3f}  {rf_mf1:>6.3f}")
    print(f"  {'CNN':<22} {cnn_acc:>7.1%}  {cnn_auc:>6.3f}  {cnn_mf1:>6.3f}")
    print(f"  {'Ensemble':<22} {ens_acc:>7.1%}  {ens_auc:>6.3f}  {ens_mf1:>6.3f}")

    os.makedirs(os.path.dirname(CM_OUT), exist_ok=True)
    plot_confusion_matrix(y_patient, y_pred, title_suffix=" (RF + CNN)")

    log_metrics({
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "pipeline":   "heart",
        "model":      "Ensemble (RF+HGB + CNN, patient-level)",
        "n_patients": len(patient_ids),
        "accuracy":   round(ens_acc, 4),
        "roc_auc":    round(ens_auc, 4),
        "mean_f1":    round(ens_mf1, 4),
        "threshold":  ens_thr,
    })


if __name__ == "__main__":
    main()
