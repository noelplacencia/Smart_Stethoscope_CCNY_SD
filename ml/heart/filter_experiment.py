"""
ml/heart/filter_experiment.py
------------------------------
Experiment: bandpass (20–950 Hz) vs lowpass (950 Hz only).

Extracts features under each filter, trains RF, compares AUC.
Nothing is saved — no model files, no metrics log touched.
"""

import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
from imblearn.over_sampling import SMOTE

DATA_DIR = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/datasets/circor/training_data"
CSV_PATH = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/datasets/circor/training_data.csv"

TARGET_SR    = 4000
WINDOW_SEC   = 3.0
HOP_SEC      = 1.5
HIGHCUT      = 950.0
BUTTER_ORDER = 6
LABEL_MAP    = {"Absent": 0, "Present": 1}


def bandpass_filter(signal, sr):
    nyq = sr / 2.0
    b, a = butter(BUTTER_ORDER, [20.0 / nyq, HIGHCUT / nyq], btype="band")
    return filtfilt(b, a, signal)


def lowpass_filter(signal, sr):
    nyq = sr / 2.0
    b, a = butter(BUTTER_ORDER, HIGHCUT / nyq, btype="low")
    return filtfilt(b, a, signal)


def _band_energy(fft_power, freqs, low, high):
    mask = (freqs >= low) & (freqs < high)
    return float(np.mean(fft_power[mask])) if mask.any() else 0.0


def _shannon_energy(window):
    x2 = window ** 2
    x2 = np.where(x2 > 0, x2, 1e-10)
    return -x2 * np.log(x2)


def extract_features(window, sr):
    mfccs       = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
    mfcc_means  = np.mean(mfccs, axis=1)
    mfcc_deltas = np.mean(librosa.feature.delta(mfccs, order=1), axis=1)
    mfcc_delta2 = np.mean(librosa.feature.delta(mfccs, order=2), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr))
    rolloff  = np.mean(librosa.feature.spectral_rolloff(y=window, sr=sr, roll_percent=0.85))
    zcr      = np.mean(librosa.feature.zero_crossing_rate(y=window))
    rms      = np.mean(librosa.feature.rms(y=window))
    fft_power = np.abs(np.fft.rfft(window)) ** 2
    freqs     = np.fft.rfftfreq(len(window), d=1.0 / sr)
    peak_freq = freqs[np.argmax(fft_power)]
    e_low   = _band_energy(fft_power, freqs, 20.0,  150.0)
    e_mid   = _band_energy(fft_power, freqs, 150.0, 500.0)
    e_high  = _band_energy(fft_power, freqs, 500.0, 950.0)
    e_ratio = e_low / (e_mid + 1e-10)
    se = _shannon_energy(window)
    return np.concatenate([
        mfcc_means, mfcc_deltas, mfcc_delta2,
        [centroid, rolloff, zcr, rms, peak_freq, np.mean(window), np.std(window)],
        [e_low, e_mid, e_high, e_ratio],
        [np.mean(se), np.std(se), np.max(se)],
    ])


def extract_all(filter_fn, label=""):
    df_meta = pd.read_csv(CSV_PATH)
    df_meta = df_meta[df_meta["Murmur"].isin(LABEL_MAP)].copy()
    all_wavs = [f for f in os.listdir(DATA_DIR) if f.endswith(".wav")]
    wav_index = {}
    for wav in all_wavs:
        pid = wav.split("_")[0]
        wav_index.setdefault(pid, []).append(wav)

    rows, labels, groups = [], [], []
    window_size = int(WINDOW_SEC * TARGET_SR)
    hop_size    = int(HOP_SEC   * TARGET_SR)

    for _, row in df_meta.iterrows():
        pid   = str(row["Patient ID"])
        label_int = LABEL_MAP[row["Murmur"]]
        for wav_name in wav_index.get(pid, []):
            path = os.path.join(DATA_DIR, wav_name)
            audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
            audio = filter_fn(audio, TARGET_SR)
            i = 0
            while i + window_size <= len(audio):
                rows.append(extract_features(audio[i:i + window_size], TARGET_SR))
                labels.append(label_int)
                groups.append(pid)
                i += hop_size

    return np.array(rows), np.array(labels), np.array(groups)


def run(X, y, groups, name):
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X[train_idx])
    X_test  = scaler.transform(X[test_idx])

    X_bal, y_bal = SMOTE(random_state=42).fit_resample(X_train, y[train_idx])

    rf = RandomForestClassifier(
        n_estimators=300, min_samples_split=4, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1,
    )
    hgb = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=6,
        min_samples_leaf=20, class_weight="balanced", random_state=42,
    )
    ensemble = VotingClassifier(
        estimators=[("rf", RandomForestClassifier(
            n_estimators=300, min_samples_split=4, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1,
        )), ("hgb", HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_depth=6,
            min_samples_leaf=20, class_weight="balanced", random_state=42,
        ))],
        voting="soft",
    )

    rf.fit(X_bal, y_bal)
    hgb.fit(X_bal, y_bal)
    ensemble.fit(X_bal, y_bal)

    rf_auc  = roc_auc_score(y[test_idx], rf.predict_proba(X_test)[:, 1])
    hgb_auc = roc_auc_score(y[test_idx], hgb.predict_proba(X_test)[:, 1])
    ens_auc = roc_auc_score(y[test_idx], ensemble.predict_proba(X_test)[:, 1])

    print(f"  {name:<30}  RF={rf_auc:.3f}  HGB={hgb_auc:.3f}  Ensemble={ens_auc:.3f}")
    return ens_auc


if __name__ == "__main__":
    print("Extracting features — bandpass (20–950 Hz) ...")
    X_bp, y_bp, g_bp = extract_all(bandpass_filter)
    print(f"  {len(X_bp)} windows\n")

    print("Extracting features — lowpass (950 Hz only) ...")
    X_lp, y_lp, g_lp = extract_all(lowpass_filter)
    print(f"  {len(X_lp)} windows\n")

    print("=" * 65)
    print(f"  {'Filter':<30}  {'RF AUC':>6}  {'HGB AUC':>7}  {'Ens AUC':>7}")
    print("=" * 65)
    run(X_bp, y_bp, g_bp, "Bandpass  20–950 Hz (current)")
    run(X_lp, y_lp, g_lp, "Lowpass   0–950 Hz (professor)")
    print("=" * 65)
