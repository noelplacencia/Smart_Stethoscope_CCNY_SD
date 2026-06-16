"""
ml/lung/visualize_patient.py
-----------------------------
4-panel visualization for a single lung test patient (ICBHI):
  1. Filtered waveform with per-window predicted class shading
  2. Log-mel spectrogram
  3. Per-window RF class probability timeline (4 lines)
  4. Bode plot of the Butterworth bandpass filter

Set PATIENT_ID = None to auto-select the first crackle patient from the
test set. Set to a string (e.g. "104") to pick a specific one.

Output: ml/data/plots/patient_visualization_lung_{patient_id}.png
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import joblib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy.signal import butter, filtfilt, freqz
from scipy.stats import kurtosis as signal_kurtosis
from sklearn.model_selection import GroupShuffleSplit

# ── Config ─────────────────────────────────────────────────────────────────────
PATIENT_ID = None  # str or None (auto-selects first crackle patient in test set)

# ── Paths ──────────────────────────────────────────────────────────────────────
ICBHI_DIR       = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/datasets/icbhi"
FEATURES_CSV    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/features_lung.csv"
FEATURES_HF_CSV = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/features_hf_lung.csv"
RF_MODEL        = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/rf_model_lung.joblib"
RF_SCALER       = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/scaler_lung.joblib"
PLOTS_DIR       = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/plots"

# ── DSP parameters (must match extract_features.py) ───────────────────────────
TARGET_SR    = 16000
WINDOW_SEC   = 3.0
HOP_SEC      = 1.5
LOWCUT       = 20.0
HIGHCUT      = 2000.0
BUTTER_ORDER = 6
N_MELS       = 64
HOP_LENGTH   = 512

LABEL_NAMES   = ["normal", "crackle", "wheeze", "both"]
CLASS_COLORS  = ["steelblue", "crimson", "darkorange", "purple"]


# ── DSP helpers (inline from extract_features.py) ─────────────────────────────

def bandpass_filter(signal, sr):
    nyq = sr / 2.0
    b, a = butter(BUTTER_ORDER, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")
    return filtfilt(b, a, signal)


def extract_features(window, sr):
    mfccs            = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
    mfcc_means       = np.mean(mfccs, axis=1)
    delta_mfcc_means = np.mean(librosa.feature.delta(mfccs), axis=1)
    centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr))
    rolloff  = np.mean(librosa.feature.spectral_rolloff(y=window, sr=sr, roll_percent=0.85))
    zcr      = np.mean(librosa.feature.zero_crossing_rate(y=window))
    rms      = np.mean(librosa.feature.rms(y=window))
    contrast = np.mean(librosa.feature.spectral_contrast(
        y=window, sr=sr, n_bands=6, fmin=200.0), axis=1)
    kurt = signal_kurtosis(window)
    return np.concatenate([
        mfcc_means, delta_mfcc_means,
        [centroid, rolloff, zcr, rms],
        contrast, [kurt],
    ])


# ── Patient selection ──────────────────────────────────────────────────────────

def get_test_patient(patient_id_override):
    dfs = [pd.read_csv(FEATURES_CSV)]
    if os.path.exists(FEATURES_HF_CSV):
        dfs.append(pd.read_csv(FEATURES_HF_CSV))
    df     = pd.concat(dfs, ignore_index=True)
    groups = df["patient_id"].astype(str).values
    X      = df.drop(columns=["label", "patient_id"]).values
    y      = df["label"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X, y, groups))
    test_pids   = groups[test_idx]
    test_labels = y[test_idx]

    # Only ICBHI patients (numeric IDs) — HF_Lung_V1 patients won't have wav files here
    icbhi_wavs = {f.split("_")[0] for f in os.listdir(ICBHI_DIR) if f.endswith(".wav")}

    if patient_id_override is not None:
        pid = str(patient_id_override)
        if pid not in test_pids:
            raise ValueError(f"Patient {pid} is not in the test set.")
        label = int(test_labels[test_pids == pid][0])
        return pid, label

    # Auto-select: first crackle (label=1) ICBHI patient in test set
    for pid, lbl in zip(test_pids, test_labels):
        if lbl == 1 and pid in icbhi_wavs:
            return str(pid), int(lbl)
    # Fallback: any ICBHI patient
    for pid, lbl in zip(test_pids, test_labels):
        if pid in icbhi_wavs:
            return str(pid), int(lbl)
    raise RuntimeError("No ICBHI patients found in test set.")


def find_wav(patient_id):
    for f in sorted(os.listdir(ICBHI_DIR)):
        if f.startswith(f"{patient_id}_") and f.endswith(".wav"):
            return os.path.join(ICBHI_DIR, f)
    raise FileNotFoundError(f"No wav found for patient {patient_id} in {ICBHI_DIR}")


# ── RF inference ───────────────────────────────────────────────────────────────

def run_rf(audio, sr, rf, scaler):
    window_size = int(WINDOW_SEC * sr)
    hop_size    = int(HOP_SEC   * sr)
    probs_all, times = [], []
    i = 0
    while i + window_size <= len(audio):
        feats = extract_features(audio[i:i + window_size], sr)
        prob  = rf.predict_proba(scaler.transform(feats.reshape(1, -1)))[0]  # (4,)
        probs_all.append(prob)
        times.append((i / sr) + WINDOW_SEC / 2)
        i += hop_size
    return np.array(times), np.array(probs_all)  # (N,), (N, 4)


# ── Bode plot ──────────────────────────────────────────────────────────────────

def bode_magnitude():
    nyq = TARGET_SR / 2.0
    b, a = butter(BUTTER_ORDER, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")
    w, h = freqz(b, a, worN=4096, fs=TARGET_SR)
    return w, 20 * np.log10(np.abs(h) + 1e-12)


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot(patient_id, audio, sr, times, probs, true_label):
    duration = len(audio) / sr
    t_audio  = np.linspace(0, duration, len(audio))

    pred_per_window = np.argmax(probs, axis=1)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"Smart Stethoscope — Lung Pipeline\n"
        f"Patient {patient_id} · True label: {LABEL_NAMES[true_label].capitalize()}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(4, 1, 4)

    # ── Panel 1: Waveform ──────────────────────────────────────────────────────
    ax1.plot(t_audio, audio, color="steelblue", linewidth=0.3, alpha=0.7)
    hop_size = int(HOP_SEC * sr)
    for idx, (pred, t_center) in enumerate(zip(pred_per_window, times)):
        t_start = t_center - WINDOW_SEC / 2
        t_end   = t_center + WINDOW_SEC / 2
        ax1.axvspan(t_start, t_end, alpha=0.2,
                    color=CLASS_COLORS[pred], linewidth=0)
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=CLASS_COLORS[i], alpha=0.5, label=LABEL_NAMES[i])
                  for i in range(4)]
    ax1.legend(handles=legend_els, loc="upper right", fontsize=8, ncol=4)
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.set_title("Filtered Waveform (20–2000 Hz bandpass) — shaded by predicted class", fontsize=10)
    ax1.set_xlim(0, duration)

    # ── Panel 2: Mel spectrogram ───────────────────────────────────────────────
    mel    = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel",
        ax=ax2, cmap="magma",
    )
    ax2.set_title("Log-Mel Spectrogram", fontsize=10)
    ax2.set_ylabel("Frequency (Hz)", fontsize=10)
    plt.colorbar(img, ax=ax2, format="%+2.0f dB", pad=0.01)

    # ── Panel 3: 4-class probability timeline ─────────────────────────────────
    for cls_idx, (name, color) in enumerate(zip(LABEL_NAMES, CLASS_COLORS)):
        ax3.plot(times, probs[:, cls_idx], label=name.capitalize(),
                 color=color, linewidth=1.5, marker="o", markersize=3)
    patient_avg = probs.mean(axis=0)
    patient_pred = int(np.argmax(patient_avg))
    correct = patient_pred == true_label
    ax3.set_title(
        f"Per-window RF class probabilities — Patient-level prediction: "
        f"{LABEL_NAMES[patient_pred].capitalize()} "
        f"({'✓ Correct' if correct else '✗ Incorrect'})",
        fontsize=10,
    )
    ax3.set_ylabel("Probability", fontsize=10)
    ax3.set_xlabel("Time (seconds)", fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=9, loc="upper right", ncol=4)

    # ── Panel 4: Bode plot ─────────────────────────────────────────────────────
    freqs_hz, mag_db = bode_magnitude()
    ax4.semilogx(freqs_hz, mag_db, color="steelblue", linewidth=1.5)
    ax4.axvline(LOWCUT,  color="crimson",    linestyle="--", linewidth=1,
                label=f"Low cutoff ({LOWCUT:.0f} Hz)")
    ax4.axvline(HIGHCUT, color="darkorange", linestyle="--", linewidth=1,
                label=f"High cutoff ({HIGHCUT:.0f} Hz)")
    ax4.set_xlim(1, TARGET_SR / 2)
    ax4.set_ylim(bottom=min(mag_db.min() - 5, -80))
    ax4.set_xlabel("Frequency (Hz)", fontsize=10)
    ax4.set_ylabel("Magnitude (dB)", fontsize=10)
    ax4.set_title(
        f"Butterworth Bandpass Filter — Order {BUTTER_ORDER} "
        f"({LOWCUT:.0f}–{HIGHCUT:.0f} Hz) — Maximally flat passband (no ripple by design)",
        fontsize=10,
    )
    ax4.legend(fontsize=9)
    ax4.grid(True, which="both", alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    os.makedirs(PLOTS_DIR, exist_ok=True)
    out = os.path.join(PLOTS_DIR, f"patient_visualization_lung_{patient_id}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    patient_id, true_label = get_test_patient(PATIENT_ID)
    wav_path = find_wav(patient_id)

    print(f"Patient   : {patient_id}")
    print(f"WAV       : {wav_path}")
    print(f"True label: {LABEL_NAMES[true_label]}")

    audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    audio    = bandpass_filter(audio, TARGET_SR)
    print(f"Duration  : {len(audio)/TARGET_SR:.1f}s  ({len(audio)} samples)")

    rf     = joblib.load(RF_MODEL)
    scaler = joblib.load(RF_SCALER)

    print("Running RF over sliding windows ...")
    times, probs = run_rf(audio, TARGET_SR, rf, scaler)
    patient_pred = LABEL_NAMES[int(np.argmax(probs.mean(axis=0)))]
    print(f"Windows   : {len(times)}  |  Patient prediction: {patient_pred}")

    plot(patient_id, audio, TARGET_SR, times, probs, true_label)


if __name__ == "__main__":
    main()
