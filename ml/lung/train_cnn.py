"""
ml/lung/train_cnn.py
Transfer learning (MobileNetV2) for ICBHI lung sound classification.
Input: log mel-spectrograms (3 × 224 × 224).
Classes: 0=normal  1=crackle  2=wheeze  3=both
"""

import json
import os
from datetime import datetime

import numpy as np
import librosa
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, f1_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from scipy.signal import butter, filtfilt

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/datasets/icbhi"
MODEL_OUT   = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/cnn_model_lung.pth"
CM_OUT      = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/plots/confusion_matrix_lung_cnn.png"
METRICS_LOG = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/metrics_log.json"
MEL_CACHE    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/mel_cache_icbhi.npz"
MEL_CACHE_HF = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/mel_cache_hf.npz"

# ── Audio/spectrogram parameters ───────────────────────────────────────────────
TARGET_SR  = 16000
WINDOW_SEC = 3.0
HOP_SEC    = 1.5
LOWCUT     = 20.0
HIGHCUT    = 2000.0
N_MELS     = 64
HOP_LENGTH = 512
IMG_SIZE   = 224

# ── Training hyperparameters ───────────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS_P1  = 15    # phase 1: classifier head only
EPOCHS_P2  = 20    # phase 2: last 4 blocks + head
LR_P1      = 1e-3
LR_P2      = 1e-4
LABEL_NAMES = ["normal", "crackle", "wheeze", "both"]


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
    return librosa.power_to_db(mel, ref=np.max)   # (N_MELS, T)


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """Focal loss: down-weights easy negatives to focus training on hard cases."""
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma  = gamma

    def forward(self, inputs, targets):
        ce  = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction="none")
        pt  = torch.exp(-ce)
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

class ICBHIDataset(Dataset):
    """Wraps shared numpy arrays + an index list — no mel data is ever copied."""
    def __init__(self, mels, labels, indices, augment=False):
        self.mels    = mels    # (N, N_MELS, T) — shared reference
        self.labels  = labels  # (N,)            — shared reference
        self.indices = indices
        self.augment = augment

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx   = self.indices[i]
        mel   = self.mels[idx].copy()   # (N_MELS, T)
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

def parse_annotation(txt_path):
    cycles = []
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            start, end = float(parts[0]), float(parts[1])
            crackle, wheeze = int(parts[2]), int(parts[3])
            label = 3 if (crackle and wheeze) else 1 if crackle else 2 if wheeze else 0
            cycles.append((start, end, label))
    return cycles


def load_all_samples():
    """
    Returns (mels, labels, patient_ids) as numpy arrays.
    Builds and caches mel spectrograms from ICBHI on first run, then merges
    HF_Lung_V1 cache if present. No Python dicts — all data stays in arrays.
    """
    if os.path.exists(MEL_CACHE):
        print(f"Loading cached mel spectrograms from {MEL_CACHE} ...")
        cache       = np.load(MEL_CACHE)
        mels        = cache["mels"]        # (N, N_MELS, T)
        labels      = cache["labels"]      # (N,)
        patient_ids = cache["patient_ids"] # (N,)
    else:
        mel_list, label_list, pat_list = [], [], []
        wav_files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".wav"))
        win_sz    = int(WINDOW_SEC * TARGET_SR)
        hop_sz    = int(HOP_SEC * TARGET_SR)

        print(f"Building mel cache from {len(wav_files)} WAV files (first time only) ...")
        for idx, wav_name in enumerate(wav_files):
            base       = wav_name.replace(".wav", "")
            wav_path   = os.path.join(DATA_DIR, wav_name)
            txt_path   = os.path.join(DATA_DIR, base + ".txt")
            patient_id = wav_name.split("_")[0]
            if not os.path.exists(txt_path):
                continue
            try:
                audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
                audio    = bandpass_filter(audio, TARGET_SR)
            except Exception as e:
                print(f"  [SKIP] {wav_name}: {e}")
                continue
            for start, end, label in parse_annotation(txt_path):
                start_s = int(start * TARGET_SR)
                end_s   = int(end   * TARGET_SR)
                cycle   = audio[start_s:end_s]
                i = 0
                while i + win_sz <= len(cycle):
                    mel_list.append(audio_to_mel(cycle[i:i + win_sz]))
                    label_list.append(label)
                    pat_list.append(patient_id)
                    i += hop_sz
            if (idx + 1) % 100 == 0:
                print(f"  {idx+1}/{len(wav_files)} files processed ...")

        mels        = np.array(mel_list,   dtype=np.float32)
        labels      = np.array(label_list, dtype=np.int32)
        patient_ids = np.array(pat_list)
        del mel_list, label_list, pat_list
        os.makedirs(os.path.dirname(MEL_CACHE), exist_ok=True)
        np.savez(MEL_CACHE, mels=mels, labels=labels, patient_ids=patient_ids)
        print(f"Cache saved to {MEL_CACHE}")

    # Merge HF_Lung_V1 mel cache if present
    if os.path.exists(MEL_CACHE_HF):
        print(f"Loading HF_Lung_V1 mel cache from {MEL_CACHE_HF} ...")
        hf          = np.load(MEL_CACHE_HF)
        mels        = np.concatenate([mels,        hf["mels"]],       axis=0)
        labels      = np.concatenate([labels,      hf["labels"]],     axis=0)
        patient_ids = np.concatenate([patient_ids, hf["patient_ids"]], axis=0)
        print(f"  Added {len(hf['mels'])} HF windows")

    n = len(mels)
    print(f"Loaded {n} windows from {len(set(patient_ids))} patients")
    for i, name in enumerate(LABEL_NAMES):
        count = int((labels == i).sum())
        print(f"  {name:<10} {count:>5}  ({count/n:.1%})")
    return mels, labels, patient_ids


# ── Model ──────────────────────────────────────────────────────────────────────

def build_model(n_classes=4):
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
    y_bin   = label_binarize(y_test, classes=[0, 1, 2, 3])
    acc     = float(np.mean(y_pred == y_test))
    auc     = float(roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro"))
    mean_f1 = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    return acc, auc, mean_f1


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_test, y_pred):
    cm     = confusion_matrix(y_test, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.set_xticks(range(len(LABEL_NAMES))); ax.set_xticklabels(LABEL_NAMES, fontsize=11)
    ax.set_yticks(range(len(LABEL_NAMES))); ax.set_yticklabels(LABEL_NAMES, fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=12)
    ax.set_ylabel("True label", fontsize=12)
    ax.set_title("Smart Stethoscope — Lung CNN (MobileNetV2)\n"
                 "(count / % of true class)", fontsize=12, pad=12)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > thresh else "navy"
            ax.text(j, i, f"{cm[i, j]}\n({cm_pct[i, j]:.1f}%)",
                    ha="center", va="center", fontsize=10, color=color)
    totals = cm.sum(axis=1)
    handles = [plt.Rectangle((0, 0), 1, 1, fc="none", ec="none") for _ in LABEL_NAMES]
    ax.legend(handles, [f"{n}: {t}" for n, t in zip(LABEL_NAMES, totals)],
              title="True class totals", loc="upper right",
              bbox_to_anchor=(1.35, 1), fontsize=9, title_fontsize=9)
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
                print("Warning: metrics_log.json was malformed — starting fresh")
                log = []
    log.append(entry)
    with open(METRICS_LOG, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Metrics logged to: {METRICS_LOG}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Smart Stethoscope — Lung CNN (Transfer Learning)")
    print("=" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    mels, labels, patient_ids = load_all_samples()

    # Patient-level 80/20 split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(mels, labels, groups=patient_ids))

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(4), y=labels[train_idx]
    )
    print(f"Class weights: {np.round(class_weights, 2)}")

    train_pats = len(set(patient_ids[train_idx]))
    test_pats  = len(set(patient_ids[test_idx]))
    print(f"\nTrain : {len(train_idx)} windows ({train_pats} patients)")
    print(f"Test  : {len(test_idx)} windows ({test_pats} patients)")

    # Datasets share the same mels/labels arrays — no copies made
    train_set = ICBHIDataset(mels, labels, train_idx, augment=True)
    test_set  = ICBHIDataset(mels, labels, test_idx,  augment=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = build_model(n_classes=4).to(device)
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
        "n_classes":        4,
        "label_names":      LABEL_NAMES,
        "img_size":         IMG_SIZE,
        "n_mels":           N_MELS,
        "hop_length":       HOP_LENGTH,
        "target_sr":        TARGET_SR,
    }, MODEL_OUT)
    print(f"\nModel saved to: {MODEL_OUT}")
    print("Copy cnn_model_lung.pth to Raspberry Pi when ready.")

    log_metrics({
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "pipeline":   "lung",
        "model":      "MobileNetV2 (transfer learning, focal loss)",
        "n_windows":  len(mels),
        "n_patients": int(len(set(patient_ids))),
        "accuracy":   round(acc,     4),
        "roc_auc":    round(auc,     4),
        "mean_f1":    round(mean_f1, 4),
    })


if __name__ == "__main__":
    main()
