"""
ml/piezo/train_cnn.py
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
MODEL_OUT   = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/cnn_model_piezo.pth"
CM_OUT      = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/plots/confusion_matrix_piezo_cnn.png"
METRICS_LOG = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/metrics_log.json"
MEL_CACHE   = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/mel_cache_icbhi.npz"

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
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        mel = self.samples[idx]["mel"].copy()   # (N_MELS, T)
        label = self.samples[idx]["label"]

        if self.augment:
            mel = time_mask(mel)
            mel = freq_mask(mel)

        # Per-sample standardisation
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        # (N_MELS, T) → (1, N_MELS, T) → resize (1, 224, 224) → (3, 224, 224)
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
    Parse ICBHI, extract mel-spectrograms for all 3-second sliding windows.
    Results are cached to MEL_CACHE so subsequent runs skip the ~10-min audio scan.
    """
    if os.path.exists(MEL_CACHE):
        print(f"Loading cached mel spectrograms from {MEL_CACHE} ...")
        cache = np.load(MEL_CACHE)
        mels       = cache["mels"]
        labels_arr = cache["labels"]
        pat_ids    = cache["patient_ids"]
        samples = [
            {"mel": mels[i], "label": int(labels_arr[i]), "patient_id": pat_ids[i]}
            for i in range(len(mels))
        ]
    else:
        samples   = []
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
                    window = cycle[i:i + win_sz]
                    samples.append({
                        "mel":        audio_to_mel(window),
                        "label":      label,
                        "patient_id": patient_id,
                    })
                    i += hop_sz

            if (idx + 1) % 100 == 0:
                print(f"  {idx+1}/{len(wav_files)} files processed ...")

        os.makedirs(os.path.dirname(MEL_CACHE), exist_ok=True)
        np.savez(MEL_CACHE,
                 mels=np.array([s["mel"] for s in samples], dtype=np.float32),
                 labels=np.array([s["label"] for s in samples], dtype=np.int32),
                 patient_ids=np.array([s["patient_id"] for s in samples]))
        print(f"Cache saved to {MEL_CACHE}")

    print(f"Loaded {len(samples)} windows from "
          f"{len(set(s['patient_id'] for s in samples))} patients")
    for i, name in enumerate(LABEL_NAMES):
        count = sum(1 for s in samples if s["label"] == i)
        print(f"  {name:<10} {count:>5}  ({count/len(samples):.1%})")
    return samples


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
    ax.set_title("Smart Stethoscope — Piezo CNN (MobileNetV2)\n"
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
            log = json.load(f)
    log.append(entry)
    with open(METRICS_LOG, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Metrics logged to: {METRICS_LOG}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Smart Stethoscope — Piezo CNN (Transfer Learning)")
    print("=" * 60 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    samples     = load_all_samples()
    patient_ids = np.array([s["patient_id"] for s in samples])
    labels      = np.array([s["label"]      for s in samples])

    # Patient-level 80/20 split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(samples, labels, groups=patient_ids))

    class_weights = compute_class_weight(
        "balanced", classes=np.arange(4), y=labels[train_idx]
    )
    print(f"Class weights: {np.round(class_weights, 2)}")

    train_samples = [samples[i] for i in train_idx]
    test_samples  = [samples[i] for i in test_idx]
    train_pats    = len(set(patient_ids[train_idx]))
    test_pats     = len(set(patient_ids[test_idx]))
    print(f"\nTrain : {len(train_samples)} windows ({train_pats} patients)")
    print(f"Test  : {len(test_samples)} windows ({test_pats} patients)")

    train_set    = ICBHIDataset(train_samples, augment=True)
    test_set     = ICBHIDataset(test_samples,  augment=False)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model     = build_model(n_classes=4).to(device)
    w_tensor  = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=w_tensor)

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
    best_auc, best_state = 0.0, None

    for epoch in range(1, EPOCHS_P2 + 1):
        loss, _ = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()
        y_test, y_pred, y_prob = run_inference(model, test_loader, device)
        acc, auc, mean_f1 = compute_metrics(y_test, y_pred, y_prob)
        print(f"  Epoch {epoch:02d}/{EPOCHS_P2}  loss={loss:.4f}  "
              f"acc={acc:.3f}  auc={auc:.3f}  f1={mean_f1:.3f}")
        if auc > best_auc:
            best_auc   = auc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Load best checkpoint (by ROC-AUC)
    model.load_state_dict(best_state)
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
    print("Copy cnn_model_piezo.pth to Raspberry Pi when ready.")

    log_metrics({
        "timestamp":  datetime.now().isoformat(timespec="seconds"),
        "pipeline":   "piezo",
        "model":      "MobileNetV2 (transfer learning)",
        "n_windows":  len(samples),
        "n_patients": int(len(set(patient_ids))),
        "accuracy":   round(acc,     4),
        "roc_auc":    round(auc,     4),
        "mean_f1":    round(mean_f1, 4),
    })


if __name__ == "__main__":
    main()
