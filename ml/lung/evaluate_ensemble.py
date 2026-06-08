"""
ml/lung/evaluate_ensemble.py
-----------------------------
Patient-level soft ensemble: RF + MobileNetV2 CNN (4-class lung).

For each patient in the test set:
  - RF side : mean of all per-class window probabilities  → argmax
  - CNN side: mean of all per-class softmax window probs  → argmax
  - Ensemble: 0.5 * RF_probs + 0.5 * CNN_probs           → argmax

Ground-truth patient label: majority class across that patient's windows.

Uses the same GroupShuffleSplit (random_state=42) as both training scripts.

Output:
    ml/data/plots/confusion_matrix_lung_ensemble.png
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
from scipy import stats as scipy_stats
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
FEATURES_CSV    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/features_lung.csv"
FEATURES_HF_CSV = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/features_hf_lung.csv"
RF_MODEL        = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/rf_model_lung.joblib"
RF_SCALER       = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/scaler_lung.joblib"
CNN_MODEL       = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/cnn_model_lung.pth"
MEL_CACHE       = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/mel_cache_icbhi.npz"
MEL_CACHE_HF    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/mel_cache_hf.npz"
CM_OUT          = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/plots/confusion_matrix_lung_ensemble.png"
METRICS_LOG     = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/metrics_log.json"

IMG_SIZE    = 224
BATCH_SIZE  = 64
NUM_CLASSES = 4
LABEL_NAMES = ["normal", "crackle", "wheeze", "both"]


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

def patient_majority_label(pid_array, label_array):
    """Return {patient_id: majority_class_label} dict."""
    from collections import Counter
    pat_labels = defaultdict(list)
    for pid, lbl in zip(pid_array, label_array):
        pat_labels[pid].append(int(lbl))
    return {pid: Counter(lbls).most_common(1)[0][0] for pid, lbls in pat_labels.items()}


def evaluate_patient_level(label, y_patient, y_probs):
    """
    y_probs: (N_patients, 4) averaged class probabilities.
    Picks argmax as predicted class.
    """
    y_pred = np.argmax(y_probs, axis=1)
    present = np.unique(y_patient)
    try:
        auc = float(roc_auc_score(
            y_patient, y_probs[:, present],
            multi_class="ovr", average="macro",
            labels=present,
        ))
    except ValueError:
        auc = float("nan")
    acc  = float(np.mean(y_pred == y_patient))
    mf1  = float(f1_score(y_patient, y_pred, average="macro", zero_division=0))
    print(f"\n{'─'*14} {label} (patient-level) {'─'*14}")
    print(classification_report(y_patient, y_pred, target_names=LABEL_NAMES, zero_division=0))
    print(f"  Accuracy : {acc:.1%}")
    print(f"  ROC-AUC  : {auc:.3f}")
    print(f"  Mean F1  : {mf1:.3f}")
    return y_pred, acc, auc, mf1


def plot_confusion_matrix(y_test, y_pred, title_suffix=""):
    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(NUM_CLASSES)); ax.set_xticklabels(LABEL_NAMES, fontsize=10)
    ax.set_yticks(range(NUM_CLASSES)); ax.set_yticklabels(LABEL_NAMES, fontsize=10)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title(f"Smart Stethoscope — Lung Ensemble{title_suffix}\n"
                 "(count / % of true class)", fontsize=12, pad=12)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "navy"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=9, color=color)
    totals = cm.sum(axis=1)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="none", ec="none") for _ in LABEL_NAMES]
    ax.legend(handles, [f"{n}: {t}" for n, t in zip(LABEL_NAMES, totals)],
              title="True class totals", loc="upper right",
              bbox_to_anchor=(1.45, 1), fontsize=9, title_fontsize=9)
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Smart Stethoscope — Lung Ensemble Evaluation")
    print("  (patient-level: RF + CNN soft vote, 4-class)")
    print("=" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── RF side ───────────────────────────────────────────────────────────────
    print("\nLoading RF features ...")
    dfs = [pd.read_csv(FEATURES_CSV)]
    if os.path.exists(FEATURES_HF_CSV):
        dfs.append(pd.read_csv(FEATURES_HF_CSV))
    df     = pd.concat(dfs, ignore_index=True)
    groups = df["patient_id"].astype(str).values
    X      = df.drop(columns=["label", "patient_id"]).values
    y      = df["label"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx   = next(gss.split(X, y, groups))
    test_patients = set(groups[test_idx])
    print(f"  Test set: {len(test_idx)} windows, {len(test_patients)} patients")

    scaler       = joblib.load(RF_SCALER)
    rf           = joblib.load(RF_MODEL)
    X_test_rf    = scaler.transform(X[test_idx])
    rf_probs_win = rf.predict_proba(X_test_rf)   # (N, 4)
    rf_pids      = groups[test_idx]
    rf_labels    = y[test_idx]

    # Aggregate RF probs per patient (sum then normalize = mean)
    patient_rf_probs  = defaultdict(lambda: np.zeros(NUM_CLASSES))
    patient_rf_counts = defaultdict(int)
    for probs, pid in zip(rf_probs_win, rf_pids):
        patient_rf_probs[pid]  += probs
        patient_rf_counts[pid] += 1

    patient_label = patient_majority_label(rf_pids, rf_labels)

    # ── CNN side ──────────────────────────────────────────────────────────────
    print("\nLoading CNN mel cache ...")
    cache = np.load(MEL_CACHE)
    mels, mel_labels, mel_pids = cache["mels"], cache["labels"], cache["patient_ids"].astype(str)
    if os.path.exists(MEL_CACHE_HF):
        hf = np.load(MEL_CACHE_HF)
        mels       = np.concatenate([mels, hf["mels"]],        axis=0)
        mel_labels = np.concatenate([mel_labels, hf["labels"]], axis=0)
        mel_pids   = np.concatenate([mel_pids, hf["patient_ids"].astype(str)], axis=0)

    test_mask = np.isin(mel_pids, list(test_patients))
    mels_test = mels[test_mask]
    pids_test = mel_pids[test_mask]
    print(f"  CNN test windows: {test_mask.sum()}")

    ckpt  = torch.load(CNN_MODEL, map_location=device)
    model = mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
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
            cnn_probs_win.append(probs)
    cnn_probs_win = np.vstack(cnn_probs_win)  # (N, 4)

    patient_cnn_probs  = defaultdict(lambda: np.zeros(NUM_CLASSES))
    patient_cnn_counts = defaultdict(int)
    for probs, pid in zip(cnn_probs_win, pids_test):
        patient_cnn_probs[pid]  += probs
        patient_cnn_counts[pid] += 1

    # ── Patient-level aggregation ─────────────────────────────────────────────
    patient_ids = sorted(set(patient_rf_probs.keys()) & set(patient_cnn_probs.keys()))
    dropped = len(test_patients) - len(patient_ids)
    if dropped:
        print(f"  ({dropped} patients dropped — no CNN windows)")

    y_patient = np.array([patient_label[p] for p in patient_ids])

    rf_pat = np.array([
        patient_rf_probs[p] / patient_rf_counts[p] for p in patient_ids
    ])  # (N_patients, 4)

    cnn_pat = np.array([
        patient_cnn_probs[p] / patient_cnn_counts[p] for p in patient_ids
    ])  # (N_patients, 4)

    ens_pat = 0.5 * rf_pat + 0.5 * cnn_pat

    print(f"\nPatients evaluated: {len(patient_ids)}")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:<10} {int((y_patient == i).sum())}")

    # ── Evaluate ──────────────────────────────────────────────────────────────
    _, rf_acc,  rf_auc,  rf_mf1  = evaluate_patient_level("RF",              y_patient, rf_pat)
    _, cnn_acc, cnn_auc, cnn_mf1 = evaluate_patient_level("CNN",             y_patient, cnn_pat)
    y_pred, ens_acc, ens_auc, ens_mf1 = evaluate_patient_level(
        "RF + CNN Ensemble", y_patient, ens_pat
    )

    print(f"\n{'='*60}")
    print(f"  Summary (patient-level)")
    print(f"{'='*60}")
    print(f"  {'Model':<22} {'Acc':>7}  {'AUC':>6}  {'F1':>6}")
    print(f"  {'─'*22}  {'─'*6}  {'─'*5}  {'─'*5}")
    print(f"  {'RF':<22} {rf_acc:>7.1%}  {rf_auc:>6.3f}  {rf_mf1:>6.3f}")
    print(f"  {'CNN':<22} {cnn_acc:>7.1%}  {cnn_auc:>6.3f}  {cnn_mf1:>6.3f}")
    print(f"  {'Ensemble':<22} {ens_acc:>7.1%}  {ens_auc:>6.3f}  {ens_mf1:>6.3f}")

    os.makedirs(os.path.dirname(CM_OUT), exist_ok=True)
    plot_confusion_matrix(y_patient, y_pred, title_suffix=" (RF + CNN)")

    log_metrics({
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "pipeline":   "lung",
        "model":      "Ensemble (RF + CNN, patient-level)",
        "n_patients": len(patient_ids),
        "accuracy":   round(ens_acc, 4),
        "roc_auc":    round(ens_auc, 4),
        "mean_f1":    round(ens_mf1, 4),
    })


if __name__ == "__main__":
    main()
