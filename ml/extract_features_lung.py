"""
extract_features_lung.py
-------------------
Loads the ICBHI 2017 dataset, applies the smart stethoscope DSP pipeline,
and extracts a 17-feature vector per 3-second window.

Output: ml/data/features_lung.csv
"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt
from scipy.stats import kurtosis as signal_kurtosis

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR  = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/ICBHI_final_database"
OUT_CSV   = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/features_lung.csv"

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
    38 features per window.

    Features:
        0-12   MFCCs 1-13 (mean)
        13-25  Delta MFCCs 1-13 (mean) — temporal dynamics, patient-invariant
        26     Spectral centroid (mean)
        27     Spectral rolloff at 85% (mean)
        28     Zero crossing rate (mean)
        29     RMS energy (mean)
        30-36  Spectral contrast bands 1-7 (mean) — peak/valley ratio per band
        37     Kurtosis — signal impulsiveness (high=crackle, low=wheeze)
    """
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
    assert len(features) == 38, f"Expected 38 features, got {len(features)}"
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
    all_patients = []
    skipped      = 0

    for i, wav_name in enumerate(sorted(wav_files)):
        base       = wav_name.replace(".wav", "")
        wav_path   = os.path.join(DATA_DIR, wav_name)
        txt_path   = os.path.join(DATA_DIR, base + ".txt")
        patient_id = wav_name.split("_")[0]

        if not os.path.exists(txt_path):
            print(f"  [SKIP] No annotation for {wav_name}")
            skipped += 1
            continue

        try:
            results = process_file(wav_path, txt_path)
            for feats, label in results:
                all_features.append(feats)
                all_labels.append(label)
                all_patients.append(patient_id)

            print(f"  [{i+1}/{len(wav_files)}] {wav_name} — {len(results)} windows")

        except Exception as e:
            print(f"  [ERROR] {wav_name}: {e}")
            skipped += 1

    # ── Save to CSV ────────────────────────────────────────────────────────────
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

    print(f"\nDone.")
    print(f"  Total windows  : {len(df)}")
    print(f"  Unique patients: {df['patient_id'].nunique()}")
    print(f"  Skipped files  : {skipped}")
    print(f"  Label counts   :\n{df['label'].value_counts().rename({v:k for k,v in LABEL_MAP.items()})}")
    print(f"  Saved to       : {OUT_CSV}")


if __name__ == "__main__":
    main()