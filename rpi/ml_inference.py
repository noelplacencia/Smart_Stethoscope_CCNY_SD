"""
rpi/ml_inference.py
Runs heart and lung CNN inference using ONNX Runtime (no PyTorch on RPi).

Preprocessing replicates train_cnn.py exactly:
  bandpass filter → log mel spectrogram → per-window z-score →
  bilinear resize to img_size×img_size → 3-channel input
"""

import os
import numpy as np
import librosa
from scipy.signal import butter, filtfilt
from scipy.ndimage import zoom
import onnxruntime as ort

_HEART_LOWCUT  = 20.0
_HEART_HIGHCUT = 950.0
_LUNG_LOWCUT   = 20.0
_LUNG_HIGHCUT  = 2000.0
_ESP32_SR      = 16000


def _audio_to_array(audio: np.ndarray, meta: dict,
                    lowcut: float, highcut: float) -> np.ndarray:
    """Return (1, 3, img_size, img_size) float32 array ready for ONNX input."""
    target_sr  = meta["target_sr"]
    n_mels     = meta["n_mels"]
    hop_length = meta["hop_length"]
    img_size   = meta["img_size"]

    if target_sr != _ESP32_SR:
        audio = librosa.resample(audio, orig_sr=_ESP32_SR, target_sr=target_sr)

    nyq = target_sr / 2.0
    b, a = butter(6, [lowcut / nyq, highcut / nyq], btype="band")
    audio = filtfilt(b, a, audio).astype(np.float32)

    mel = librosa.feature.melspectrogram(
        y=audio, sr=target_sr, n_mels=n_mels,
        hop_length=hop_length, fmin=lowcut, fmax=highcut,
    )
    mel = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

    mel = (mel - mel.mean()) / (mel.std() + 1e-8)

    # Resize to img_size × img_size
    sy = img_size / mel.shape[0]
    sx = img_size / mel.shape[1]
    mel = zoom(mel, (sy, sx), order=1).astype(np.float32)

    mel = np.stack([mel, mel, mel], axis=0)   # (3, H, W)
    return mel[np.newaxis]                     # (1, 3, H, W)


def _load_onnx(path: str):
    """Return (ort.InferenceSession, meta_dict) or (None, None)."""
    if not os.path.exists(path):
        return None, None

    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])

    json_path = path.replace(".onnx", ".json")
    if not os.path.exists(json_path):
        print(f"[inference] WARNING: metadata JSON not found at {json_path}")
        return None, None

    import json
    with open(json_path) as f:
        meta = json.load(f)

    return sess, meta


class InferenceEngine:
    def __init__(self, heart_model_path: str, lung_model_path: str):
        heart_onnx = heart_model_path.replace(".pth", ".onnx")
        lung_onnx  = lung_model_path.replace(".pth",  ".onnx")

        self._heart_sess, self._heart_meta = _load_onnx(heart_onnx)
        self._lung_sess,  self._lung_meta  = _load_onnx(lung_onnx)

        if self._heart_sess:
            print(f"[inference] heart model loaded "
                  f"({self._heart_meta['n_classes']} classes)")
        else:
            print(f"[inference] WARNING: heart ONNX not found at {heart_onnx}")

        if self._lung_sess:
            print(f"[inference] lung model loaded "
                  f"({self._lung_meta['n_classes']} classes)")
        else:
            print(f"[inference] WARNING: lung ONNX not found at {lung_onnx}")

    def run(self, audio: np.ndarray, mode: str = "heart") -> dict:
        if mode == "heart":
            sess, meta = self._heart_sess, self._heart_meta
            lowcut, highcut = _HEART_LOWCUT, _HEART_HIGHCUT
        else:
            sess, meta = self._lung_sess, self._lung_meta
            lowcut, highcut = _LUNG_LOWCUT, _LUNG_HIGHCUT

        if sess is None or meta is None:
            return {"mode": mode, "label": "unavailable",
                    "confidence": 0.0, "probabilities": {}}

        try:
            inp = _audio_to_array(audio, meta, lowcut, highcut)
            input_name = sess.get_inputs()[0].name
            logits = sess.run(None, {input_name: inp})[0][0]
            exp = np.exp(logits - logits.max())
            probs = exp / exp.sum()

            pred_idx   = int(np.argmax(probs))
            label      = meta["label_names"][pred_idx]
            confidence = float(probs[pred_idx])

            return {
                "mode":          mode,
                "label":         label,
                "confidence":    round(confidence, 3),
                "probabilities": {
                    name: round(float(p), 3)
                    for name, p in zip(meta["label_names"], probs)
                },
            }
        except Exception as e:
            print(f"[inference] error during {mode} inference: {e}")
            return {"mode": mode, "label": "error",
                    "confidence": 0.0, "probabilities": {}}
