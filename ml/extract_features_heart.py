"""
extract_features_heart.py
-------------------------
Loads the CirCor DigiScope dataset, applies DSP tuned for heart sounds,
and extracts a 17-feature vector per 3-second window.

Feature vector matches extract_features.py exactly so both lung and heart
data can feed the same training pipeline.

Dataset layout expected (download from PhysioNet):
    ml/data/circor-heart-sound/
        training_data.csv       ← patient metadata + murmur labels
        training_data/          ← {patient_id}_{location}.wav files

Labels (patient-level, from training_data.csv "Murmur" column):
    Absent  → 0
    Present → 1
    Unknown → skipped

Output: ml/data/features_heart.csv
"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/circor-heart-sound/training_data"
CSV_PATH = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/circor-heart-sound/training_data.csv"
OUT_CSV  = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/features_heart.csv"

# ── DSP parameters ─────────────────────────────────────────────────────────────
TARGET_SR    = 4000    # CirCor native rate; heart sounds are <500 Hz
WINDOW_SEC   = 3.0
HOP_SEC      = 1.5     # 50 % overlap
LOWCUT       = 20.0
HIGHCUT      = 950.0   # well below Nyquist (2000 Hz at 4 kHz SR)
BUTTER_ORDER = 6

# ── Label map ──────────────────────────────────────────────────────────────────
LABEL_MAP  = {"Absent": 0, "Present": 1}
LABEL_NAMES = ["absent", "present"]


def bandpass_filter(signal: np.ndarray, sr: int) -> np.ndarray:
    """Zero-phase Butterworth bandpass (20–950 Hz) for heart sounds."""
    nyq = sr / 2.0
    b, a = butter(BUTTER_ORDER, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")
    return filtfilt(b, a, signal)


def extract_features(window: np.ndarray, sr: int) -> np.ndarray:
    """
    17 features — identical order to extract_features.py (lung pipeline).

    Features:
        0-12  MFCCs 1-13 (mean across window)
        13    Spectral centroid (mean)
        14    Spectral rolloff at 85% (mean)
        15    Zero crossing rate (mean)
        16    RMS energy (mean)
    """
    mfccs      = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)

    centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr))
    rolloff  = np.mean(librosa.feature.spectral_rolloff(y=window, sr=sr, roll_percent=0.85))
    zcr      = np.mean(librosa.feature.zero_crossing_rate(y=window))
    rms      = np.mean(librosa.feature.rms(y=window))

    features = np.concatenate([mfcc_means, [centroid, rolloff, zcr, rms]])
    assert len(features) == 17, f"Expected 17 features, got {len(features)}"
    return features


def process_file(wav_path: str, label_int: int):
    """
    Load one recording, filter, slide 3-second windows, extract features.
    Returns list of (feature_vector, label_int).
    """
    audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    audio = bandpass_filter(audio, TARGET_SR)

    window_size = int(WINDOW_SEC * TARGET_SR)
    hop_size    = int(HOP_SEC   * TARGET_SR)

    results = []
    i = 0
    while i + window_size <= len(audio):
        feats = extract_features(audio[i : i + window_size], TARGET_SR)
        results.append((feats, label_int))
        i += hop_size
    return results


def main():
    print(f"Loading patient metadata: {CSV_PATH}")
    df_meta = pd.read_csv(CSV_PATH)

    df_meta = df_meta[df_meta["Murmur"].isin(LABEL_MAP)].copy()
    n_present = (df_meta["Murmur"] == "Present").sum()
    n_absent  = (df_meta["Murmur"] == "Absent").sum()
    print(f"{len(df_meta)} patients with known label  "
          f"(present={n_present}, absent={n_absent})\n")

    # Index wav files by patient ID prefix
    all_wavs  = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
    wav_index: dict[str, list[str]] = {}
    for wav in all_wavs:
        pid = wav.split("_")[0]
        wav_index.setdefault(pid, []).append(wav)

    all_features, all_labels = [], []
    skipped = 0
    total_files = 0

    for _, row in df_meta.iterrows():
        pid       = str(int(row["Patient ID"]))
        label_int = LABEL_MAP[row["Murmur"]]
        wavs      = wav_index.get(pid, [])

        if not wavs:
            print(f"  [SKIP] No wav files for patient {pid}")
            skipped += 1
            continue

        for wav_name in sorted(wavs):
            wav_path = os.path.join(DATA_DIR, wav_name)
            try:
                results = process_file(wav_path, label_int)
                for feats, lbl in results:
                    all_features.append(feats)
                    all_labels.append(lbl)
                print(f"  {wav_name:<30} {len(results):>3} windows  [{row['Murmur']}]")
                total_files += 1
            except Exception as e:
                print(f"  [ERROR] {wav_name}: {e}")
                skipped += 1

    # ── Save ───────────────────────────────────────────────────────────────────
    col_names = [f"mfcc_{i+1}" for i in range(13)] + \
                ["spectral_centroid", "spectral_rolloff", "zcr", "rms"]

    df_out = pd.DataFrame(all_features, columns=col_names)
    df_out["label"] = all_labels
    df_out.to_csv(OUT_CSV, index=False)

    print(f"\nDone.")
    print(f"  Recordings processed : {total_files}")
    print(f"  Total windows        : {len(df_out)}")
    print(f"  Skipped              : {skipped}")
    print(f"  Label counts         :\n"
          f"{df_out['label'].value_counts().rename({v: k for k, v in LABEL_MAP.items()})}")
    print(f"  Saved to             : {OUT_CSV}")


if __name__ == "__main__":
    main()
