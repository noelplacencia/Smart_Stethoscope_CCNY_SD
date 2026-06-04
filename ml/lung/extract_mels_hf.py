"""
lung/extract_mels_hf.py
-----------------------
Extracts mel-spectrograms from HF_Lung_V1 (train + test) using the same
DSP pipeline as train_cnn.py, saved as mel_cache_hf.npz so both
lung and piezo CNN scripts can load and merge it with the ICBHI cache.

Label mapping (HF → ICBHI-compatible):
    No adventitious sounds          → normal  (0)
    D only                          → crackle (1)
    Wheeze / Rhonchi / Stridor only → wheeze  (2)
    D + any wheeze-type             → both    (3)

Output: ml/data/mel_cache_hf.npz  (mels, labels, patient_ids)
"""

import os
import re
import numpy as np
import librosa
from scipy.signal import butter, filtfilt

# ── Paths ──────────────────────────────────────────────────────────────────────
HF_DIRS = [
    "/home/noel/Smart_Stethoscope_CCNY_SD/ml/datasets/hf_lung_v1/train",
    "/home/noel/Smart_Stethoscope_CCNY_SD/ml/datasets/hf_lung_v1/test",
]
OUT_CACHE = "/home/noel/Smart_Stethoscope_CCNY_SD/ml/data/mel_cache_hf.npz"

# ── DSP parameters (must match train_cnn.py) ──────────────────────────────────
TARGET_SR    = 16000
WINDOW_SEC   = 3.0
HOP_SEC      = 1.5
LOWCUT       = 20.0
HIGHCUT      = 2000.0
BUTTER_ORDER = 6
N_MELS       = 64
HOP_LENGTH   = 512

WHEEZE_TYPES = {"wheeze", "rhonchi", "stridor"}


def bandpass_filter(signal, sr):
    nyq = sr / 2.0
    b, a = butter(BUTTER_ORDER, [LOWCUT / nyq, HIGHCUT / nyq], btype="band")
    return filtfilt(b, a, signal)


def audio_to_mel(audio):
    mel = librosa.feature.melspectrogram(
        y=audio, sr=TARGET_SR, n_mels=N_MELS,
        hop_length=HOP_LENGTH, fmin=LOWCUT, fmax=HIGHCUT,
    )
    return librosa.power_to_db(mel, ref=np.max)


def parse_hf_time(t: str) -> float:
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def parse_hf_label(txt_path: str):
    crackle_intervals = []
    wheeze_intervals  = []
    with open(txt_path) as f:
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
    return any(s < win_end and e > win_start for s, e in intervals)


def patient_id_from_filename(wav_name: str) -> str:
    if wav_name.startswith("steth_"):
        m = re.match(r"steth_(\d{8})", wav_name)
        return f"hf_steth_{m.group(1)}" if m else f"hf_{wav_name}"
    elif wav_name.startswith("trunc_"):
        m = re.match(r"trunc_(\d{4}-\d{2}-\d{2})", wav_name)
        return f"hf_trunc_{m.group(1)}" if m else f"hf_{wav_name}"
    return f"hf_{wav_name}"


def main():
    print("=" * 60)
    print("  HF_Lung_V1 Mel Spectrogram Extraction")
    print("=" * 60)

    all_mels, all_labels, all_patients = [], [], []
    total_files = skipped = 0
    win_sz = int(WINDOW_SEC * TARGET_SR)
    hop_sz = int(HOP_SEC    * TARGET_SR)

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
                audio, _ = librosa.load(wav_path, sr=TARGET_SR, mono=True)
                audio    = bandpass_filter(audio, TARGET_SR)
                crackle_ivs, wheeze_ivs = parse_hf_label(txt_path)

                i = 0
                while i + win_sz <= len(audio):
                    win_start = i / TARGET_SR
                    win_end   = (i + win_sz) / TARGET_SR
                    window    = audio[i : i + win_sz]

                    has_crackle = overlaps(win_start, win_end, crackle_ivs)
                    has_wheeze  = overlaps(win_start, win_end, wheeze_ivs)
                    if has_crackle and has_wheeze:
                        label = 3
                    elif has_crackle:
                        label = 1
                    elif has_wheeze:
                        label = 2
                    else:
                        label = 0

                    all_mels.append(audio_to_mel(window))
                    all_labels.append(label)
                    all_patients.append(pat_id)
                    i += hop_sz

            except Exception as e:
                print(f"  [ERROR] {wav_name}: {e}")
                skipped += 1

        print(f"  Done. Running total: {len(all_mels)} windows")

    mels_arr = np.array(all_mels,    dtype=np.float32)
    labs_arr = np.array(all_labels,  dtype=np.int32)
    pats_arr = np.array(all_patients)

    os.makedirs(os.path.dirname(OUT_CACHE), exist_ok=True)
    np.savez(OUT_CACHE, mels=mels_arr, labels=labs_arr, patient_ids=pats_arr)

    label_names = {0: "normal", 1: "crackle", 2: "wheeze", 3: "both"}
    print(f"\nTotal files     : {total_files}")
    print(f"Skipped         : {skipped}")
    print(f"Total windows   : {len(all_mels)}")
    pat_set = set(all_patients)
    print(f"Unique patients : {len(pat_set)}")
    print("Class distribution:")
    for k, v in label_names.items():
        count = int((labs_arr == k).sum())
        print(f"  {v:<10} {count:>6}  ({count/len(labs_arr):.1%})")
    print(f"\nSaved to: {OUT_CACHE}")


if __name__ == "__main__":
    main()
