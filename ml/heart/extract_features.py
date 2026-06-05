"""
extract_features_heart.py
-------------------------
Loads the CirCor DigiScope dataset, applies DSP tuned for heart sounds,
and extracts a 53-feature vector per 3-second window.

Dataset layout expected (download from PhysioNet):
    ml/datasets/circor/
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
DATA_DIR = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/datasets/circor/training_data"
CSV_PATH = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/datasets/circor/training_data.csv"
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


def _band_energy(fft_power: np.ndarray, freqs: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    return float(np.mean(fft_power[mask])) if mask.any() else 0.0


def _shannon_energy(window: np.ndarray) -> np.ndarray:
    x2 = window ** 2
    x2 = np.where(x2 > 0, x2, 1e-10)
    return -x2 * np.log(x2)


def extract_features(window: np.ndarray, sr: int) -> np.ndarray:
    """
    53 features per window.

    Features:
        0-12   MFCCs 1-13 (mean)
        13-25  MFCC deltas 1-13 (mean)
        26-38  MFCC delta-deltas 1-13 (mean)
        39     Spectral centroid (mean)
        40     Spectral rolloff at 85% (mean)
        41     Zero crossing rate (mean)
        42     RMS energy (mean)
        43     Peak frequency (dominant FFT bin)
        44     Mean amplitude
        45     Std amplitude
        46     Band energy 20-150 Hz (S1/S2 range)
        47     Band energy 150-500 Hz (murmur range)
        48     Band energy 500-950 Hz (high)
        49     Band energy ratio low/mid (46/47)
        50     Shannon energy envelope: mean
        51     Shannon energy envelope: std
        52     Shannon energy envelope: max
    """
    mfccs        = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
    mfcc_means   = np.mean(mfccs, axis=1)
    mfcc_deltas  = np.mean(librosa.feature.delta(mfccs, order=1), axis=1)
    mfcc_delta2  = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)

    centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr))
    rolloff  = np.mean(librosa.feature.spectral_rolloff(y=window, sr=sr, roll_percent=0.85))
    zcr      = np.mean(librosa.feature.zero_crossing_rate(y=window))
    rms      = np.mean(librosa.feature.rms(y=window))

    fft_power = np.abs(np.fft.rfft(window)) ** 2
    freqs     = np.fft.rfftfreq(len(window), d=1.0 / sr)
    peak_freq = freqs[np.argmax(fft_power)]
    mean_val  = np.mean(window)
    std_val   = np.std(window)

    e_low  = _band_energy(fft_power, freqs, 20.0,  150.0)
    e_mid  = _band_energy(fft_power, freqs, 150.0, 500.0)
    e_high = _band_energy(fft_power, freqs, 500.0, 950.0)
    e_ratio = e_low / (e_mid + 1e-10)

    se      = _shannon_energy(window)
    se_mean = float(np.mean(se))
    se_std  = float(np.std(se))
    se_max  = float(np.max(se))

    features = np.concatenate([
        mfcc_means, mfcc_deltas, mfcc_delta2,
        [centroid, rolloff, zcr, rms, peak_freq, mean_val, std_val],
        [e_low, e_mid, e_high, e_ratio],
        [se_mean, se_std, se_max],
    ])
    assert len(features) == 53, f"Expected 53 features, got {len(features)}"
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

    all_features, all_labels, all_patient_ids = [], [], []
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
                    all_patient_ids.append(pid)
                print(f"  {wav_name:<30} {len(results):>3} windows  [{row['Murmur']}]")
                total_files += 1
            except Exception as e:
                print(f"  [ERROR] {wav_name}: {e}")
                skipped += 1

    # ── Save ───────────────────────────────────────────────────────────────────
    col_names = (
        [f"mfcc_{i+1}"       for i in range(13)] +
        [f"mfcc_d_{i+1}"     for i in range(13)] +
        [f"mfcc_d2_{i+1}"    for i in range(13)] +
        ["spectral_centroid", "spectral_rolloff", "zcr", "rms",
         "peak_frequency", "mean", "std"] +
        ["band_e_low", "band_e_mid", "band_e_high", "band_e_ratio"] +
        ["shannon_mean", "shannon_std", "shannon_max"]
    )

    df_out = pd.DataFrame(all_features, columns=col_names)
    df_out["label"]      = all_labels
    df_out["patient_id"] = all_patient_ids
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
