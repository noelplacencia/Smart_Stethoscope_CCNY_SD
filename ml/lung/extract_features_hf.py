"""
lung/extract_features_hf.py
---------------------------
Extracts the same 38-feature vectors from HF_Lung_V1 (train + test folders)
using the same DSP pipeline as extract_features.py so the features are
directly comparable and can be concatenated with features_lung.csv.

Label mapping (HF → ICBHI-compatible):
    No adventitious sounds          → normal  (0)
    D only                          → crackle (1)
    Wheeze / Rhonchi / Stridor only → wheeze  (2)
    D + any wheeze-type             → both    (3)

Patient ID: date portion of filename
    steth_YYYYMMDD_HH_MM_ss  → hf_steth_YYYYMMDD
    trunc_YYYY-MM-DD-...     → hf_trunc_YYYY-MM-DD

Output: ml/data/features_hf_lung.csv
"""

import os
import re
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis as signal_kurtosis

# ── Paths ──────────────────────────────────────────────────────────────────────
HF_DIRS  = [
    "/home/noel/Smart_Stethoscope_CCNY_SD/ml/datasets/hf_lung_v1/train",
    "/home/noel/Smart_Stethoscope_CCNY_SD/ml/datasets/hf_lung_v1/test",
]
OUT_CSV  = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/features_hf_lung.csv"

# ── DSP parameters (must match extract_features.py) ───────────────────────────
TARGET_SR    = 16000
WINDOW_SEC   = 3.0
HOP_SEC      = 1.5
LOWCUT       = 20.0
HIGHCUT      = 2000.0
BUTTER_ORDER = 6

WHEEZE_TYPES = {"wheeze", "rhonchi", "stridor"}


def bandpass_filter(signal: np.ndarray, sr: int) -> np.ndarray:
    nyq  = sr / 2.0
    b, a = butter(BUTTER_ORDER, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")
    return filtfilt(b, a, signal)


def extract_features(window: np.ndarray, sr: int) -> np.ndarray:
    mfccs            = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
    mfcc_means       = np.mean(mfccs, axis=1)
    delta_mfcc_means = np.mean(librosa.feature.delta(mfccs), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr))
    rolloff  = np.mean(librosa.feature.spectral_rolloff(y=window, sr=sr, roll_percent=0.85))
    zcr      = np.mean(librosa.feature.zero_crossing_rate(y=window))
    rms      = np.mean(librosa.feature.rms(y=window))
    contrast = np.mean(librosa.feature.spectral_contrast(y=window, sr=sr, n_bands=6, fmin=200.0), axis=1)
    kurt     = signal_kurtosis(window)
    features = np.concatenate([mfcc_means, delta_mfcc_means,
                                [centroid, rolloff, zcr, rms],
                                contrast, [kurt]])
    assert len(features) == 38
    return features


def parse_hf_time(t: str) -> float:
    """Convert HH:MM:SS.mmm to seconds."""
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_hf_label(txt_path: str):
    """
    Returns two lists of (start_sec, end_sec) intervals:
        crackle_intervals, wheeze_intervals
    Ignores I (inhalation) and E (exhalation) lines.
    """
    crackle_intervals = []
    wheeze_intervals  = []

    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            event = parts[0].lower()
            try:
                start = parse_hf_time(parts[1])
                end   = parse_hf_time(parts[2])
            except (ValueError, IndexError):
                continue

            if event == "d":
                crackle_intervals.append((start, end))
            elif event in WHEEZE_TYPES:
                wheeze_intervals.append((start, end))

    return crackle_intervals, wheeze_intervals


def overlaps(win_start, win_end, intervals):
    """True if any interval overlaps [win_start, win_end)."""
    for s, e in intervals:
        if s < win_end and e > win_start:
            return True
    return False


def patient_id_from_filename(wav_name: str) -> str:
    """Extract a stable patient-level ID from the filename."""
    if wav_name.startswith("steth_"):
        # steth_YYYYMMDD_HH_MM_ss.wav
        m = re.match(r"steth_(\d{8})", wav_name)
        return f"hf_steth_{m.group(1)}" if m else f"hf_{wav_name}"
    elif wav_name.startswith("trunc_"):
        # trunc_YYYY-MM-DD-HH-MM-ss-LX_N.wav
        m = re.match(r"trunc_(\d{4}-\d{2}-\d{2})", wav_name)
        return f"hf_trunc_{m.group(1)}" if m else f"hf_{wav_name}"
    return f"hf_{wav_name}"


def process_file(wav_path: str, txt_path: str):
    audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    audio    = bandpass_filter(audio, TARGET_SR)

    crackle_ivs, wheeze_ivs = parse_hf_label(txt_path)

    window_size = int(WINDOW_SEC * TARGET_SR)
    hop_size    = int(HOP_SEC    * TARGET_SR)
    results     = []

    i = 0
    while i + window_size <= len(audio):
        win_start = i / TARGET_SR
        win_end   = (i + window_size) / TARGET_SR
        window    = audio[i : i + window_size]

        has_crackle = overlaps(win_start, win_end, crackle_ivs)
        has_wheeze  = overlaps(win_start, win_end, wheeze_ivs)

        if has_crackle and has_wheeze:
            label = 3  # both
        elif has_crackle:
            label = 1  # crackle
        elif has_wheeze:
            label = 2  # wheeze
        else:
            label = 0  # normal

        feats = extract_features(window, TARGET_SR)
        results.append((feats, label))
        i += hop_size

    return results


def main():
    print("=" * 60)
    print("  HF_Lung_V1 Feature Extraction")
    print("=" * 60)

    all_features, all_labels, all_patients = [], [], []
    total_files = skipped = 0

    for folder in HF_DIRS:
        wav_files = sorted(f for f in os.listdir(folder) if f.endswith(".wav"))
        print(f"\nFolder: {folder}  ({len(wav_files)} WAV files)")

        for wav_name in wav_files:
            base     = wav_name.replace(".wav", "")
            wav_path = os.path.join(folder, wav_name)
            txt_path = os.path.join(folder, base + "_label.txt")
            pat_id   = patient_id_from_filename(wav_name)
            total_files += 1

            if not os.path.exists(txt_path):
                skipped += 1
                continue

            try:
                results = process_file(wav_path, txt_path)
                for feats, label in results:
                    all_features.append(feats)
                    all_labels.append(label)
                    all_patients.append(pat_id)
            except Exception as e:
                print(f"  [ERROR] {wav_name}: {e}")
                skipped += 1

        print(f"  Done. Running total: {len(all_features)} windows")

    col_names = (
        [f"mfcc_{i+1}"       for i in range(13)] +
        [f"delta_mfcc_{i+1}" for i in range(13)] +
        ["spectral_centroid", "spectral_rolloff", "zcr", "rms"] +
        [f"spectral_contrast_{i+1}" for i in range(7)] +
        ["kurtosis"]
    )

    df = pd.DataFrame(all_features, columns=col_names)
    df["label"]      = all_labels
    df["patient_id"] = all_patients
    df.to_csv(OUT_CSV, index=False)

    label_names = {0: "normal", 1: "crackle", 2: "wheeze", 3: "both"}
    print(f"\nTotal files     : {total_files}")
    print(f"Skipped         : {skipped}")
    print(f"Total windows   : {len(df)}")
    print(f"Unique patients : {df['patient_id'].nunique()}")
    print("Class distribution:")
    for k, v in label_names.items():
        count = (df["label"] == k).sum()
        print(f"  {v:<10} {count:>6}  ({count/len(df):.1%})")
    print(f"Saved to: {OUT_CSV}")


if __name__ == "__main__":
    main()
