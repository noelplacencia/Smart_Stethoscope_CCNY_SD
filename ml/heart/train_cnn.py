"""
ml/heart/train_cnn.py
Transfer learning (MobileNetV2) for CirCor heart murmur detection.
Input: log mel-spectrograms (3 × 224 × 224).
Classes: 0=absent  1=present
"""

import json
import os
from datetime import datetime

import numpy as np
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from scipy.signal import butter, filtfilt

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/datasets/circor/training_data"
CSV_PATH    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/datasets/circor/training_data.csv"
MODEL_OUT   = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/cnn_model_heart.pth"
CM_OUT      = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/plots/confusion_matrix_heart_cnn.png"
METRICS_LOG = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/metrics_log.json"
MEL_CACHE   = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/mel_cache_heart_5s.npz"

# ── Audio/spectrogram parameters ───────────────────────────────────────────────
TARGET_SR  = 4000    # CirCor native rate; heart sounds < 500 Hz
WINDOW_SEC = 5.0     # 5s guarantees 4-5 full cardiac cycles per window
HOP_SEC    = 2.5
LOWCUT     = 20.0
HIGHCUT    = 950.0
N_MELS     = 64
HOP_LENGTH = 128     # at 4 kHz gives ~94 time frames (same resolution as lung CNN)
IMG_SIZE   = 224

# ── Training hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE  = 32
EPOCHS_P1   = 15
EPOCHS_P2   = 20
LR_P1       = 1e-3
LR_P2       = 1e-4
LABEL_MAP   = {"Absent": 0, "Present": 1}
LABEL_NAMES = ["absent", "present"]


# ── Audio helpers ──────────────────────────────────────────────────────────────

def bandpass_filter(signal, sr):
    nyq = sr / 2.0
    b, a = butter(6, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")
    return filtfilt(b, a, signal)


def audio_to_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR, n_mels=N_MELS,
        hop_length=HOP_LENGTH, fmin=LOWCUT, fmax=HIGHCUT,
    )
    return librosa.power_to_db(mel, ref=np.max)


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss: down-weights easy negatives to focus training on hard cases."""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, inputs, targets):
        ce   = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt   = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── SpecAugment ────────────────────────────────────────────────────────────────

def time_mask(mel, max_t=15):
    T = mel.shape[1]
    t = np.random.randint(1, max_t + 1)
    t0 = np.random.randint(0, max(1, T - t))
    mel[:, t0:t0 + t] = mel.min()
    return mel


def freq_mask(mel, max_f=8):
    F = mel.shape[0]
    f = np.random.randint(1, max_f + 1)
    f0 = np.random.randint(0, max(1, F - f))
    mel[f0:f0 + f, :] = mel.min()
    return mel


# ── Dataset ────────────────────────────────────────────────────────────────────

class HeartDataset(Dataset):
    """Wraps shared numpy arrays + an index list — no mel data is ever copied."""
    def __init__(self, mels, labels, indices, augment=False):
        self.mels    = mels
        self.labels  = labels
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx   = self.indices[i]
        mel   = self.mels[idx].copy()
        label = int(self.labels[idx])

        if self.augment:
            mel = time_mask(mel)
            mel = freq_mask(mel)

        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        t = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t = torch.nn.functional.interpolate(
            t, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )
        t = t.squeeze(0).repeat(3, 1, 1)

        return t, label


# ── Data loading ───────────────────────────────────────────────────────────────

def load_all_samples():
    """
    Returns (mels, labels, patient_ids) as numpy arrays.
    Builds and caches mel spectrograms from CirCor on first run.
    """
    if os.path.exists(MEL_CACHE):
        print(f"Loading cached mel spectrograms from {MEL_CACHE} ...")
        cache       = np.load(MEL_CACHE)
        mels        = cache["mels"]
        labels      = cache["labels"]
        patient_ids = cache["patient_ids"]
    else:
        print("Building mel cache from CirCor dataset (first run only) ...")
        df_meta = pd.read_csv(CSV_PATH)
        df_meta = df_meta[df_meta["Murmur"].isin(LABEL_MAP)].copy()
        n_present = (df_meta["Murmur"] == "Present").sum()
        n_absent  = (df_meta["Murmur"] == "Absent").sum()
        print(f"  {len(df_meta)} patients  (present={n_present}, absent={n_absent})")

        all_wavs  = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
        wav_index = {}
        for wav in all_wavs:
            pid = wav.split("_")[0]
            wav_index.setdefault(pid, []).append(wav)

        win_sz = int(WINDOW_SEC * TARGET_SR)
        hop_sz = int(HOP_SEC    * TARGET_SR)

        mel_list, label_list, pat_list = [], [], []
        skipped = 0

        for _, row in df_meta.iterrows():
            pid       = str(int(row["Patient ID"]))
            label_int = LABEL_MAP[row["Murmur"]]
            wavs      = wav_index.get(pid, [])
            if not wavs:
                skipped += 1
                continue
            for wav_name in sorted(wavs):
                wav_path = os.path.join(DATA_DIR, wav_name)
                try:
                    audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
                    audio    = bandpass_filter(audio, TARGET_SR)
                    i = 0
                    while i + win_sz <= len(audio):
                        mel_list.append(audio_to_mel(audio[i:i + win_sz]))
                        label_list.append(label_int)
                        pat_list.append(pid)
                        i += hop_sz
                except Exception as e:
                    print(f"  [ERROR] {wav_name}: {e}")
                    skipped += 1

        print(f"  Skipped: {skipped}")
        mels        = np.array(mel_list,   dtype=np.float32)
        labels      = np.array(label_list, dtype=np.int32)
        patient_ids = np.array(pat_list)
        del mel_list, label_list, pat_list
        os.makedirs(os.path.dirname(MEL_CACHE), exist_ok=True)
        np.savez(MEL_CACHE, mels=mels, labels=labels, patient_ids=patient_ids)
        print(f"Cache saved to {MEL_CACHE}")

    n = len(mels)
    print(f"Loaded {n} windows from {len(set(patient_ids))} patients")
    for i, name in enumerate(LABEL_NAMES):
        count = int((labels == i).sum())
        print(f"  {name:<10} {count:>5}  ({count/n:.1%})")
    return mels, labels, patient_ids


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(n_classes=2):
    model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, n_classes)
    return model


def freeze_features(model):
    for param in model.features.parameters():
        param.requires_grad = False


def unfreeze_last_blocks(model, n=4):
    blocks = list(model.features.children())
    for block in blocks[-n:]:
        for param in block.parameters():
            param.requires_grad = True


# ── Training loop ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(targets)
        correct    += (outputs.argmax(1) == targets).sum().item()
        n          += len(targets)
    return total_loss / n, correct / n


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_probs, all_targets = [], [], []
    for inputs, targets in loader:
        inputs  = inputs.to(device)
        outputs = model(inputs)
        probs   = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_preds.extend(outputs.argmax(1).cpu().numpy())
        all_targets.extend(targets.numpy())
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def compute_metrics(y_test, y_pred, y_prob):
    acc     = float(np.mean(y_pred == y_test))
    auc     = float(roc_auc_score(y_test, y_prob[:, 1]))
    mean_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    return acc, auc, mean_f1


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred):
    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(LABEL_NAMES))); ax.set_xticklabels(LABEL_NAMES, fontsize=11)
    ax.set_yticks(range(len(LABEL_NAMES))); ax.set_yticklabels(LABEL_NAMES, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Smart Stethoscope — Heart CNN (MobileNetV2)\n"
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


# ── Metrics log ────────────────────────────────────────────────────────────────

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
    print("  Smart Stethoscope — Heart CNN (Transfer Learning)")
    print("=" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    mels, labels, patient_ids = load_all_samples()

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(mels, labels, groups=patient_ids))

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(2), y=labels[train_idx]
    )
    print(f"Class weights: {np.round(class_weights, 2)}")

    train_pats = len(set(patient_ids[train_idx]))
    test_pats  = len(set(patient_ids[test_idx]))
    print(f"\nTrain : {len(train_idx)} windows ({train_pats} patients)")
    print(f"Test  : {len(test_idx)} windows ({test_pats} patients)")

    train_set    = HeartDataset(mels, labels, train_idx, augment=True)
    test_set     = HeartDataset(mels, labels, test_idx,  augment=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = build_model(n_classes=2).to(device)
    w_tensor  = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = FocalLoss(weight=w_tensor, gamma=2.0)

    # ── Phase 1: classifier head only ─────────────────────────────────────────
    print(f"\n── Phase 1: classifier head ({EPOCHS_P1} epochs) ──────────────")
    freeze_features(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_P1
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_P1)
    for epoch in range(1, EPOCHS_P1 + 1):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        print(f"  Epoch {epoch:02d}/{EPOCHS_P1}  loss={loss:.4f}  train_acc={acc:.3f}")

    # ── Phase 2: last 4 feature blocks + head ─────────────────────────────────
    print(f"\n── Phase 2: fine-tune last 4 blocks ({EPOCHS_P2} epochs) ──────")
    unfreeze_last_blocks(model, n=4)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR_P2
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_P2)
    best_auc = 0.0

    for epoch in range(1, EPOCHS_P2 + 1):
        loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        y_test, y_pred, y_prob = run_inference(model, test_loader, device)
        acc, auc, mean_f1 = compute_metrics(y_test, y_pred, y_prob)
        print(f"  Epoch {epoch:02d}/{EPOCHS_P2}  loss={loss:.4f}  "
              f"acc={acc:.3f}  auc={auc:.3f}  f1={mean_f1:.3f}")
        if auc > best_auc:
            best_auc = auc
            torch.save({k: v.clone() for k, v in model.state_dict().items()}, MODEL_OUT)
            print(f"    ↑ new best — checkpoint saved")

    model.load_state_dict(torch.load(MODEL_OUT, map_location=device))
    print(f"\nBest ROC-AUC during phase 2: {best_auc:.3f}")

    # ── Final evaluation ───────────────────────────────────────────────────────
    print("\n── Final Evaluation ──────────────────────────────────────────")
    y_test, y_pred, y_prob = run_inference(model, test_loader, device)
    acc, auc, mean_f1 = compute_metrics(y_test, y_pred, y_prob)
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES, zero_division=0))
    print(f"  Accuracy : {acc:.1%}")
    print(f"  ROC-AUC  : {auc:.3f}")
    print(f"  Mean F1  : {mean_f1:.3f}")

    os.makedirs(os.path.dirname(CM_OUT), exist_ok=True)
    plot_confusion_matrix(y_test, y_pred)

    # ── Save ───────────────────────────────────────────────────────────────────
    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture":     "mobilenet_v2",
        "n_classes":        2,
        "label_names":      LABEL_NAMES,
        "img_size":         IMG_SIZE,
        "n_mels":           N_MELS,
        "hop_length":       HOP_LENGTH,
        "target_sr":        TARGET_SR,
    }, MODEL_OUT)
    print(f"\nModel saved to: {MODEL_OUT}")
    print("Copy cnn_model_heart.pth to Raspberry Pi when ready.")

    log_metrics({
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "pipeline":   "heart",
        "model":      "MobileNetV2 (transfer learning)",
        "n_windows":  len(mels),
        "n_patients": int(len(set(patient_ids))),
        "accuracy":   round(acc,     4),
        "roc_auc":    round(auc,     4),
        "mean_f1":    round(mean_f1, 4),
    })


if __name__ == "__main__":
    main()
