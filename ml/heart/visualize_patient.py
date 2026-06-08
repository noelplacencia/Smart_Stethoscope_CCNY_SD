"""
ml/heart/visualize_patient.py
------------------------------
4-panel visualization for a single heart test patient:
  1. Filtered waveform with per-window probability shading
  2. Log-mel spectrogram
  3. Per-window RF murmur probability timeline
  4. Bode plot of the Butterworth bandpass filter

Set PATIENT_ID = None to auto-select the first murmur-positive patient
from the test set. Set to an int (e.g. 12345) to pick a specific one.
Set LOCATION to "AV"/"MV"/"PV"/"TV" or None to auto-select.

Output: ml/data/plots/patient_visualization_heart_{patient_id}_{location}.png
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
from matplotlib.patches import Patch
from scipy.signal import butter, filtfilt, freqz
from scipy.stats import kurtosis as signal_kurtosis
from sklearn.model_selection import GroupShuffleSplit

# ── Config (override these to pick a specific patient) ─────────────────────────
PATIENT_ID = None   # int or None (auto-selects first Present in test set)
LOCATION   = "AV"  # "AV" | "MV" | "PV" | "TV" | None (auto)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR     = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/datasets/circor/training_data"
CSV_PATH     = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/datasets/circor/training_data.csv"
FEATURES_CSV = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/features_heart.csv"
RF_MODEL     = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/rf_model_heart.joblib"
RF_SCALER    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/scaler_heart.joblib"
THRESHOLD    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/threshold_heart.joblib"
PLOTS_DIR    = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/plots"

# ── DSP parameters (must match extract_features.py) ───────────────────────────
TARGET_SR    = 4000
WINDOW_SEC   = 3.0
HOP_SEC      = 1.5
LOWCUT       = 20.0
HIGHCUT      = 950.0
BUTTER_ORDER = 6
N_MELS       = 64
HOP_LENGTH   = 128

LABEL_NAMES = ["absent", "present"]
LABEL_MAP   = {"Absent": 0, "Present": 1}


# ── DSP helpers (inline from extract_features.py) ─────────────────────────────

def bandpass_filter(signal, sr):
    nyq = sr / 2.0
    b, a = butter(BUTTER_ORDER, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")
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
    freqs_arr = np.fft.rfftfreq(len(window), d=1.0 / sr)
    peak_freq = freqs_arr[np.argmax(fft_power)]
    e_low   = _band_energy(fft_power, freqs_arr, 20.0,  150.0)
    e_mid   = _band_energy(fft_power, freqs_arr, 150.0, 500.0)
    e_high  = _band_energy(fft_power, freqs_arr, 500.0, 950.0)
    e_ratio = e_low / (e_mid + 1e-10)
    se = _shannon_energy(window)
    return np.concatenate([
        mfcc_means, mfcc_deltas, mfcc_delta2,
        [centroid, rolloff, zcr, rms, peak_freq, np.mean(window), np.std(window)],
        [e_low, e_mid, e_high, e_ratio],
        [np.mean(se), np.std(se), np.max(se)],
    ])


# ── Patient selection ──────────────────────────────────────────────────────────

def get_test_patient(patient_id_override):
    df = pd.read_csv(FEATURES_CSV)
    groups = df["patient_id"].values
    X = df.drop(columns=["label", "patient_id"]).values
    y = df["label"].values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, test_idx = next(gss.split(X, y, groups))
    test_pids = groups[test_idx]
    test_labels = y[test_idx]

    if patient_id_override is not None:
        pid = int(patient_id_override)
        if pid not in test_pids:
            raise ValueError(f"Patient {pid} is not in the test set.")
        label = int(test_labels[test_pids == pid][0])
        return pid, label

    # Auto-select: first Present patient in test set
    for pid, lbl in zip(test_pids, test_labels):
        if lbl == 1:
            return int(pid), int(lbl)
    # Fallback to first patient
    return int(test_pids[0]), int(test_labels[0])


def find_wav(patient_id, location_pref):
    order = [location_pref] if location_pref else ["AV", "MV", "PV", "TV"]
    for loc in order:
        path = os.path.join(DATA_DIR, f"{patient_id}_{loc}.wav")
        if os.path.exists(path):
            return path, loc
    # Fallback: any wav for this patient
    for f in sorted(os.listdir(DATA_DIR)):
        if f.startswith(f"{patient_id}_") and f.endswith(".wav"):
            loc = f.split("_")[1].split(".")[0]
            return os.path.join(DATA_DIR, f), loc
    raise FileNotFoundError(f"No wav found for patient {patient_id}")


# ── RF inference over sliding windows ─────────────────────────────────────────

def run_rf(audio, sr, rf, scaler):
    window_size = int(WINDOW_SEC * sr)
    hop_size    = int(HOP_SEC   * sr)
    probs, times = [], []
    i = 0
    while i + window_size <= len(audio):
        feats = extract_features(audio[i:i + window_size], sr)
        prob  = rf.predict_proba(scaler.transform(feats.reshape(1, -1)))[0, 1]
        probs.append(prob)
        times.append((i / sr) + WINDOW_SEC / 2)
        i += hop_size
    return np.array(times), np.array(probs)


# ── Bode plot data ─────────────────────────────────────────────────────────────

def bode_magnitude():
    nyq = TARGET_SR / 2.0
    b, a = butter(BUTTER_ORDER, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")
    w, h = freqz(b, a, worN=4096, fs=TARGET_SR)
    magnitude_db = 20 * np.log10(np.abs(h) + 1e-12)
    return w, magnitude_db


# ── Plot ───────────────────────────────────────────────────────────────────────

def plot(patient_id, location, audio, sr, times, probs, threshold, true_label):
    duration = len(audio) / sr
    t_audio  = np.linspace(0, duration, len(audio))

    cmap   = cm.get_cmap("RdYlBu_r")
    norm   = mcolors.Normalize(vmin=0, vmax=1)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"Smart Stethoscope — Heart Pipeline\n"
        f"Patient {patient_id} · {location} recording · "
        f"True label: {'Murmur Present' if true_label == 1 else 'Murmur Absent'}",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # Shared time axis for panels 1–3
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(4, 1, 4)

    # ── Panel 1: Waveform ──────────────────────────────────────────────────────
    ax1.plot(t_audio, audio, color="steelblue", linewidth=0.4, alpha=0.8)
    window_size = int(WINDOW_SEC * sr)
    hop_size    = int(HOP_SEC   * sr)
    i = 0
    for prob in probs:
        t_start = i / sr
        t_end   = t_start + WINDOW_SEC
        color   = cmap(norm(prob))
        ax1.axvspan(t_start, t_end, alpha=0.25, color=color, linewidth=0)
        i += hop_size
    ax1.set_ylabel("Amplitude", fontsize=10)
    ax1.set_title("Filtered Waveform (20–950 Hz bandpass) — shaded by P(murmur)", fontsize=10)
    ax1.set_xlim(0, duration)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax1, orientation="vertical", pad=0.01, label="P(murmur)")

    # ── Panel 2: Mel spectrogram ───────────────────────────────────────────────
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    img = librosa.display.specshow(
        mel_db, sr=sr, hop_length=HOP_LENGTH, x_axis="time", y_axis="mel",
        ax=ax2, cmap="magma",
    )
    ax2.set_title("Log-Mel Spectrogram", fontsize=10)
    ax2.set_ylabel("Frequency (Hz)", fontsize=10)
    plt.colorbar(img, ax=ax2, format="%+2.0f dB", pad=0.01)

    # ── Panel 3: Probability timeline ─────────────────────────────────────────
    bar_colors = [cmap(norm(p)) for p in probs]
    ax3.bar(times, probs, width=HOP_SEC * 0.9, color=bar_colors, align="center")
    ax3.axhline(threshold, color="crimson", linestyle="--", linewidth=1.2,
                label=f"Threshold ({threshold:.2f})")
    patient_avg = float(np.mean(probs))
    ax3.axhline(patient_avg, color="navy", linestyle=":", linewidth=1.2,
                label=f"Patient avg ({patient_avg:.2f})")
    patient_pred = 1 if patient_avg >= threshold else 0
    correct = patient_pred == true_label
    ax3.set_title(
        f"Per-window RF P(murmur) — Patient-level prediction: "
        f"{'Present' if patient_pred == 1 else 'Absent'} "
        f"({'✓ Correct' if correct else '✗ Incorrect'})",
        fontsize=10,
    )
    ax3.set_ylabel("P(murmur)", fontsize=10)
    ax3.set_xlabel("Time (seconds)", fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.legend(fontsize=9, loc="upper right")

    # ── Panel 4: Bode plot ─────────────────────────────────────────────────────
    freqs_hz, mag_db = bode_magnitude()
    ax4.semilogx(freqs_hz, mag_db, color="steelblue", linewidth=1.5)
    ax4.axvline(LOWCUT,  color="crimson", linestyle="--", linewidth=1,
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
    out = os.path.join(PLOTS_DIR, f"patient_visualization_heart_{patient_id}_{location}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    patient_id, true_label = get_test_patient(PATIENT_ID)
    wav_path, location = find_wav(patient_id, LOCATION)

    print(f"Patient   : {patient_id}")
    print(f"Location  : {location}")
    print(f"WAV       : {wav_path}")
    print(f"True label: {LABEL_NAMES[true_label]}")

    audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
    audio    = bandpass_filter(audio, TARGET_SR)
    print(f"Duration  : {len(audio)/TARGET_SR:.1f}s  ({len(audio)} samples)")

    rf      = joblib.load(RF_MODEL)
    scaler  = joblib.load(RF_SCALER)
    thr     = joblib.load(THRESHOLD)

    print("Running RF over sliding windows ...")
    times, probs = run_rf(audio, TARGET_SR, rf, scaler)
    print(f"Windows   : {len(probs)}  |  Patient avg P(murmur): {np.mean(probs):.3f}")

    plot(patient_id, location, audio, TARGET_SR, times, probs, thr, true_label)


if __name__ == "__main__":
    main()
