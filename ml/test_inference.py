"""
ml/test_inference.py
Test ONNX inference pipeline end-to-end using a real heart/lung wav file.
Simulates what the RPi receives: audio resampled to 16 kHz float32.

Usage (laptop):
    python ml/test_inference.py

Usage (RPi — after scp):
    python rpi/test_inference.py heart_sample.wav
"""

import sys
import os
import numpy as np
import librosa

# Allow running from repo root or rpi/ dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rpi"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../rpi"))

from ml_inference import InferenceEngine

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
HEART_PATH = os.path.join(DATA_DIR, "cnn_model_heart.onnx")
LUNG_PATH  = os.path.join(DATA_DIR, "cnn_model_lung.onnx")

# Default sample files — override via CLI arg
HEART_SAMPLE = os.path.join(os.path.dirname(__file__),
                            "datasets/circor/training_data/61610_PV.wav")
LUNG_SAMPLE  = os.path.join(os.path.dirname(__file__),
                            "datasets/icbhi/160_1b3_Al_mc_AKGC417L.wav")


def load_as_esp32(wav_path: str, duration_sec: float = 5.0) -> np.ndarray:
    """Load wav, resample to 16 kHz (ESP32 output rate), return float32 array."""
    audio, sr = librosa.load(wav_path, sr=16000, mono=True, duration=duration_sec)
    print(f"  loaded {os.path.basename(wav_path)}  "
          f"({len(audio)/16000:.1f}s @ 16kHz, peak={np.abs(audio).max():.3f})")
    return audio


def run_test(wav_path: str, mode: str, engine: InferenceEngine):
    duration = 5.0 if mode == "heart" else 3.0
    audio = load_as_esp32(wav_path, duration_sec=duration)
    result = engine.run(audio, mode=mode)
    print(f"  result  → label={result['label']}  "
          f"confidence={result['confidence']:.1%}")
    print(f"  probs   → {result['probabilities']}")


if __name__ == "__main__":
    print("Loading models...")
    engine = InferenceEngine(HEART_PATH, LUNG_PATH)

    print("\n── Heart inference ──────────────────────────────")
    heart_wav = sys.argv[1] if len(sys.argv) > 1 else HEART_SAMPLE
    run_test(heart_wav, "heart", engine)

    print("\n── Lung inference ───────────────────────────────")
    lung_wav = sys.argv[2] if len(sys.argv) > 2 else LUNG_SAMPLE
    run_test(lung_wav, "lung", engine)
