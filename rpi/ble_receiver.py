import asyncio
import json
import random
import math
import time
import numpy as np
from threading import Thread, Lock
from collections import deque

from bleak import BleakClient, BleakScanner

from config import (
    USE_BLE, BLE_DEVICE_NAME,
    CHARACTERISTIC_UUID, AUDIO_CHAR_UUID,
    MAX_POINTS, AUDIO_SAMPLE_RATE, AUDIO_WINDOW_SEC,
    INFERENCE_MODE, HEART_MODEL_PATH, LUNG_MODEL_PATH,
)
from ai_alerts import analyze_vitals
from ml_inference import InferenceEngine

AUDIO_WINDOW_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_WINDOW_SEC)  # 5 s × 16 kHz = 80 000

# Inference runs every INFERENCE_INTERVAL_SEC; librosa+PyTorch on RPi 4 takes ~0.5–1 s
INFERENCE_INTERVAL_SEC = 3.0


class DataStore:
    def __init__(self):
        self.lock       = Lock()
        self.latest     = {}
        self.ecg_wave   = deque(maxlen=MAX_POINTS)
        self.heart_wave = deque(maxlen=MAX_POINTS)
        self.spo2_trend = deque(maxlen=MAX_POINTS)
        self.resp_trend = deque(maxlen=MAX_POINTS)
        self.hr_trend   = deque(maxlen=MAX_POINTS)
        self.time_axis  = deque(maxlen=MAX_POINTS)

        # Rolling 5-second audio buffer for ML inference (covers both heart 5s and lung 3s)
        self.audio_buf  = deque(maxlen=AUDIO_WINDOW_SAMPLES)
        self.audio_lock = Lock()

        # Latest ML inference result (updated by inference thread)
        self._ml_result      = None
        self._ml_result_lock = Lock()

    def update(self, packet):
        with self.lock:
            packet["timestamp"] = time.strftime("%H:%M:%S")

            # Merge vitals-threshold alerts with the latest ML result
            with self._ml_result_lock:
                ml = self._ml_result
            packet["ai_result"] = analyze_vitals(packet, ml)

            self.latest = packet
            self.ecg_wave.append(packet.get("ecg", 0))
            hr = packet.get("ecg_hr") or packet.get("hr_ppg") or packet.get("heart_rate", 0)
            self.heart_wave.append(packet.get("heart_sound", 0))
            self.spo2_trend.append(packet.get("spo2", 0))
            self.resp_trend.append(packet.get("respiration", 0))
            self.hr_trend.append(hr)
            self.time_axis.append(packet["timestamp"])

    def add_audio(self, data: bytes):
        """Buffer incoming int16 little-endian BLE audio bytes."""
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        with self.audio_lock:
            self.audio_buf.extend(samples.tolist())

    def get_audio_window(self, sec: float = None):
        """
        Return a float32 numpy array of the last `sec` seconds of audio.
        Uses the full buffer if sec is None.  Returns None if not enough data.
        """
        n = AUDIO_WINDOW_SAMPLES if sec is None else int(AUDIO_SAMPLE_RATE * sec)
        with self.audio_lock:
            if len(self.audio_buf) >= n:
                return np.array(list(self.audio_buf)[-n:], dtype=np.float32)
        return None

    def set_ml_result(self, result: dict):
        with self._ml_result_lock:
            self._ml_result = result

    def get_dashboard_data(self):
        with self.lock:
            return {
                "latest":     self.latest,
                "ecg_wave":   list(self.ecg_wave),
                "heart_wave": list(self.heart_wave),
                "spo2_trend": list(self.spo2_trend),
                "resp_trend": list(self.resp_trend),
                "hr_trend":   list(self.hr_trend),
                "time_axis":  list(self.time_axis),
            }


store = DataStore()

# ── Inference thread ──────────────────────────────────────────────────────────

def _inference_loop(engine: InferenceEngine):
    """Background thread: runs CNN inference every INFERENCE_INTERVAL_SEC."""
    window_sec = 5.0 if INFERENCE_MODE == "heart" else 3.0
    while True:
        time.sleep(INFERENCE_INTERVAL_SEC)
        audio = store.get_audio_window(sec=window_sec)
        if audio is None:
            continue
        result = engine.run(audio, mode=INFERENCE_MODE)
        store.set_ml_result(result)
        print(f"[inference] {result['mode']}: {result['label']} "
              f"({result['confidence']:.0%})")


# ── Simulation mode ───────────────────────────────────────────────────────────

def generate_simulated_packet(seq):
    t = time.time()
    ecg = int(2050 + 550 * math.sin(2 * math.pi * 1.2 * t))
    if seq % 20 == 0:
        ecg += 1200

    return {
        "seq":           seq,
        "ecg":           ecg,
        "ecg_hr":        round(72 + 5 * math.sin(t * 0.5), 1),
        "heart_sound":   int(1800 + 600 * math.sin(2 * math.pi * 2.0 * t)
                             + random.randint(-100, 100)),
        "spo2":          random.randint(96, 99),
        "spo2_valid":    1,
        "hr_ppg":        random.randint(70, 80),
        "hr_ppg_valid":  1,
        "respiration":   random.randint(14, 18),
        "temperature":   round(random.uniform(36.5, 37.0), 2),
        "imu_x":         round(random.uniform(-0.05, 0.05), 3),
        "imu_y":         round(random.uniform(-0.05, 0.05), 3),
        "imu_z":         round(random.uniform(0.95, 1.05), 3),
    }


def simulation_loop():
    seq = 0
    # Feed simulated white-noise audio so inference still runs in sim mode
    rng = np.random.default_rng(0)
    audio_timer = time.time()
    while True:
        store.update(generate_simulated_packet(seq))
        seq += 1
        # Push a small chunk of fake audio at ~16 ms intervals to keep buffer warm
        if time.time() - audio_timer >= 0.016:
            chunk = rng.uniform(-0.01, 0.01, 256).astype(np.float32)
            store.add_audio((chunk * 32767).astype(np.int16).tobytes())
            audio_timer = time.time()
        time.sleep(0.05)


# ── BLE mode ──────────────────────────────────────────────────────────────────

async def ble_loop():
    await asyncio.sleep(3.0)  # give ESP32 time to restart advertising after a previous disconnect
    print("Scanning for ESP32 BLE device...")
    device = None

    while device is None:
        found = await BleakScanner.discover(timeout=8.0)
        for d in found:
            if d.name == BLE_DEVICE_NAME:
                device = d
                break
        if device is None:
            print("ESP32 not found. Retrying...")

    print(f"Found: {device.name} [{device.address}]")

    async with BleakClient(device.address) as client:
        try:
            await client.request_mtu(512)
        except Exception:
            pass

        print(f"Connected (MTU={client.mtu_size})")

        def vitals_handler(sender, data):
            try:
                packet = json.loads(data.decode("utf-8"))
                store.update(packet)
            except Exception as e:
                print("Vitals parse error:", e)

        def audio_handler(sender, data):
            store.add_audio(bytes(data))

        await client.start_notify(CHARACTERISTIC_UUID, vitals_handler)
        try:
            await client.start_notify(AUDIO_CHAR_UUID, audio_handler)
            print("Subscribed to vitals and audio characteristics.")
        except Exception:
            print("Audio characteristic not found — vitals only (flash new firmware for audio)")

        while True:
            await asyncio.sleep(1)


def start_data_receiver():
    # Start inference engine in background regardless of BLE/sim mode
    engine = InferenceEngine(HEART_MODEL_PATH, LUNG_MODEL_PATH)
    Thread(target=_inference_loop, args=(engine,), daemon=True).start()

    if USE_BLE:
        Thread(target=lambda: asyncio.run(ble_loop()), daemon=True).start()
    else:
        Thread(target=simulation_loop, daemon=True).start()
