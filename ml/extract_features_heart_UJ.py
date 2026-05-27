# ============================================
# Heart Sound Feature Extraction - PhysioNet 2016
# ============================================

import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, filtfilt

BASE_DIR = "physionet.org/files/challenge-2016/1.0.0"
OUT_CSV = "heart_features.csv"

TRAINING_FOLDERS = ["training-a"]

TARGET_SR = 2000
WINDOW_SEC = 3.0
HOP_SEC = 1.5
LOWCUT = 20.0
HIGHCUT = 600.0
BUTTER_ORDER = 6


def bandpass_filter(signal, sr):
    nyq = sr / 2.0
    low = LOWCUT / nyq
    high = HIGHCUT / nyq
    b, a = butter(BUTTER_ORDER, [low, high], btype="band")
    return filtfilt(b, a, signal)


def extract_features(window, sr):
    mfccs = librosa.feature.mfcc(y=window, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)

    centroid = np.mean(librosa.feature.spectral_centroid(y=window, sr=sr))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=window, sr=sr, roll_percent=0.85))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=window))
    rms = np.mean(librosa.feature.rms(y=window))

    fft_vals = np.abs(np.fft.rfft(window))
    freqs = np.fft.rfftfreq(len(window), d=1/sr)
    peak_freq = freqs[np.argmax(fft_vals)]

    mean_val = np.mean(window)
    std_val = np.std(window)

    features = np.concatenate([
        mfcc_means,
        [centroid, rolloff, zcr, rms, peak_freq, mean_val, std_val]
    ])

    return features


def read_reference_csv(folder_path):
    ref_path = os.path.join(folder_path, "REFERENCE.csv")
    df = pd.read_csv(ref_path, header=None, names=["file_id", "label"])

    label_dict = {}

    for _, row in df.iterrows():
        file_id = str(row["file_id"]).strip()
        label = int(row["label"])

        if label == -1:
            label_name = "normal"
        else:
            label_name = "abnormal"

        label_dict[file_id] = label_name

    return label_dict


def main():
    all_features = []
    all_labels = []
    all_files = []

    for folder in TRAINING_FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        print("Processing folder:", folder_path)

        labels = read_reference_csv(folder_path)

        wav_files = sorted([f for f in os.listdir(folder_path) if f.endswith(".wav")])
        print("Found WAV files:", len(wav_files))

        for wav_name in wav_files:
            file_id = wav_name.replace(".wav", "")

            if file_id not in labels:
                print("Skipping, no label:", wav_name)
                continue

            wav_path = os.path.join(folder_path, wav_name)

            try:
                audio, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)
                audio = bandpass_filter(audio, TARGET_SR)

                window_size = int(WINDOW_SEC * TARGET_SR)
                hop_size = int(HOP_SEC * TARGET_SR)

                if len(audio) < window_size:
                    continue

                start = 0
                while start + window_size <= len(audio):
                    window = audio[start:start + window_size]
                    feats = extract_features(window, TARGET_SR)

                    all_features.append(feats)
                    all_labels.append(labels[file_id])
                    all_files.append(wav_name)

                    start += hop_size

            except Exception as e:
                print("Error with", wav_name, ":", e)

    columns = [f"mfcc_{i+1}" for i in range(13)] + [
        "spectral_centroid",
        "spectral_rolloff",
        "zcr",
        "rms",
        "peak_frequency",
        "mean",
        "std"
    ]

    features_df = pd.DataFrame(all_features, columns=columns)
    features_df["label"] = all_labels
    features_df["file"] = all_files

    features_df.to_csv(OUT_CSV, index=False)

    print("\nFeature extraction complete.")
    print("Total feature rows:", len(features_df))
    print("Saved file:", OUT_CSV)
    print("\nLabel counts:")
    print(features_df["label"].value_counts())


if __name__ == "__main__":
    main()
