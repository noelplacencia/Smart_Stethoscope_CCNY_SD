# Smart Stethoscope Configuration

import os

# False = test dashboard without ESP32.
# True = receive real BLE packets from ESP32-S3.
USE_BLE = True

BLE_DEVICE_NAME = "SmartStethoscope_ESP32"

SERVICE_UUID        = "7b4d0001-8a7b-4d2b-9b41-000000000001"
CHARACTERISTIC_UUID = "7b4d0002-8a7b-4d2b-9b41-000000000001"  # vitals JSON (20 Hz)
AUDIO_CHAR_UUID     = "7b4d0003-8a7b-4d2b-9b41-000000000002"  # int16 PCM audio (16 kHz)

AUDIO_SAMPLE_RATE = 16000
# 5 s covers the heart CNN window (5 s); lung CNN only needs 3 s
AUDIO_WINDOW_SEC  = 5.0

HOST = "0.0.0.0"
PORT = 5000
MAX_POINTS = 250

# ── ML inference ──────────────────────────────────────────────────────────────

# "heart" → murmur detection (CirCor, binary)
# "lung"  → respiratory sound classification (ICBHI + HF_Lung_V1, 4-class)
INFERENCE_MODE = "heart"

_REPO_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODEL_DIR      = os.path.join(_REPO_ROOT, "ml", "data")
HEART_MODEL_PATH = os.path.join(_MODEL_DIR, "cnn_model_heart.pth")
LUNG_MODEL_PATH  = os.path.join(_MODEL_DIR, "cnn_model_lung.pth")
