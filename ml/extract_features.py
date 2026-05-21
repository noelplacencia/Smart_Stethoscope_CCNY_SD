"""
extract_features.py
-------------------
Loads the ICBHI 2017 dataset, applies the smart stethoscope DSP pipeline,
and extracts a 17-feature vector per 3-second window.

Output: ml/data/features.csv
"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/ICBHI_final_database"
OUT_CSV   = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/features.csv"

# ── DSP parameters ─────────────────────────────────────────────────────────────
TARGET_SR   = 16000   # resample everything to 16 kHz
WINDOW_SEC  = 3.0     # 3-second windows
HOP_SEC     = 1.5     # 50% overlap
LOWCUT      = 20.0    # bandpass low cutoff (Hz)
HIGHCUT     = 2000.0  # bandpass high cutoff (Hz)
BUTTER_ORDER = 6      # Butterworth filter order

# ── Label map ──────────────────────────────────────────────────────────────────
# ICBHI annotation files mark each cycle as:
# normal / crackle / wheeze / both (crackle + wheeze)
LABEL_MAP = {"normal": 0, "crackle": 1, "wheeze": 2, "both": 3}


def bandpass_filter(signal: np.ndarray, sr: int) -> np.ndarray:
    """Zero-phase 6th-order Butterworth bandpass filter (20–2000 Hz)."""
    nyq = sr / 2.0
    low  = LOWCUT  / nyq
    high = HIGHCUT / nyq
    b, a = butter(BUTTER_ORDER, [low, high], btype="band")
    return filtfilt(b, a, signal)   # filtfilt = zero phase (no time shift)


def extract_features(window: np.ndarray, sr: int) -> np.ndarray:
    """
    Extract 17 features from a single audio window.
    Must match the exact order implemented later on the ESP32.

    Features:
        0-12  — MFCCs 1-13 (mean across window)
        13    — Spectral centroid (mean)
        14    — Spectral rolloff (mean)
        15    — Zero crossing rate (mean)
        16    — RMS energy (mean)
    """
    # MFCCs (13 coefficients, mean across time frames)
    mfccs = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)  # shape (13,)

    # Spectral centroid
    centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr))

    # Spectral rolloff (85% of energy)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=window, sr=sr, roll_percent=0.85))

    # Zero crossing rate
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=window))

    # RMS energy
    rms = np.mean(librosa.feature.rms(y=window))

    features = np.concatenate([mfcc_means, [centroid, rolloff, zcr, rms]])
    assert len(features) == 17, f"Expected 17 features, got {len(features)}"
    return features


def parse_annotation(txt_path: str):
    """
    Parse an ICBHI annotation .txt file.
    Each line: start_time  end_time  crackle  wheeze
    Returns list of (start, end, label_str)
    """
    cycles = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            start    = float(parts[0])
            end      = float(parts[1])
            crackle  = int(parts[2])
            wheeze   = int(parts[3])

            if crackle and wheeze:
                label = "both"
            elif crackle:
                label = "crackle"
            elif wheeze:
                label = "wheeze"
            else:
                label = "normal"

            cycles.append((start, end, label))
    return cycles


def process_file(wav_path: str, txt_path: str):
    """
    Process one recording:
    - Load and resample audio
    - Apply bandpass filter
    - Slice into annotated respiratory cycles
    - Extract features from 3-second windows within each cycle
    Returns list of (feature_vector, label_int)
    """
    audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    audio = bandpass_filter(audio, TARGET_SR)

    cycles = parse_annotation(txt_path)
    window_size = int(WINDOW_SEC * TARGET_SR)
    hop_size    = int(HOP_SEC   * TARGET_SR)

    results = []
    for (start, end, label) in cycles:
        start_sample = int(start * TARGET_SR)
        end_sample   = int(end   * TARGET_SR)
        cycle_audio  = audio[start_sample:end_sample]

        # Slide window across the cycle
        i = 0
        while i + window_size <= len(cycle_audio):
            window = cycle_audio[i : i + window_size]
            feats  = extract_features(window, TARGET_SR)
            results.append((feats, LABEL_MAP[label]))
            i += hop_size

    return results


def main():
    print(f"Scanning dataset at: {DATA_DIR}")

    wav_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
    print(f"Found {len(wav_files)} WAV files\n")

    all_features = []
    all_labels   = []
    skipped      = 0

    for i, wav_name in enumerate(sorted(wav_files)):
        base     = wav_name.replace(".wav", "")
        wav_path = os.path.join(DATA_DIR, wav_name)
        txt_path = os.path.join(DATA_DIR, base + ".txt")

        if not os.path.exists(txt_path):
            print(f"  [SKIP] No annotation for {wav_name}")
            skipped += 1
            continue

        try:
            results = process_file(wav_path, txt_path)
            for feats, label in results:
                all_features.append(feats)
                all_labels.append(label)

            print(f"  [{i+1}/{len(wav_files)}] {wav_name} — {len(results)} windows")

        except Exception as e:
            print(f"  [ERROR] {wav_name}: {e}")
            skipped += 1

    # ── Save to CSV ────────────────────────────────────────────────────────────
    col_names = [f"mfcc_{i+1}" for i in range(13)] + \
                ["spectral_centroid", "spectral_rolloff", "zcr", "rms"]

    df = pd.DataFrame(all_features, columns=col_names)
    df["label"] = all_labels

    df.to_csv(OUT_CSV, index=False)

    print(f"\nDone.")
    print(f"  Total windows : {len(df)}")
    print(f"  Skipped files : {skipped}")
    print(f"  Label counts  :\n{df['label'].value_counts().rename({v:k for k,v in LABEL_MAP.items()})}")
    print(f"  Saved to      : {OUT_CSV}")


if __name__ == "__main__":
    main()