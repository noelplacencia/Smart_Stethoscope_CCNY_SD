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
)
from ai_alerts import analyze_vitals

AUDIO_WINDOW_SAMPLES = int(AUDIO_SAMPLE_RATE * AUDIO_WINDOW_SEC)  # 48 000


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

        # Rolling 3-second audio buffer for ML inference
        self.audio_buf  = deque(maxlen=AUDIO_WINDOW_SAMPLES)
        self.audio_lock = Lock()

    def update(self, packet):
        with self.lock:
            packet["timestamp"] = time.strftime("%H:%M:%S")
            packet["ai_result"] = analyze_vitals(packet)

            self.latest = packet
            self.ecg_wave.append(packet.get("ecg", 0))
            # Prefer ecg_hr from real R-peak detector; fall back to hr_ppg
            hr = packet.get("ecg_hr") or packet.get("hr_ppg") or packet.get("heart_rate", 0)
            self.heart_wave.append(packet.get("piezo", 0))
            self.spo2_trend.append(packet.get("spo2", 0))
            self.resp_trend.append(packet.get("respiration", 0))
            self.hr_trend.append(hr)
            self.time_axis.append(packet["timestamp"])

    def add_audio(self, data: bytes):
        """Buffer incoming int16 little-endian BLE audio bytes."""
        samples = np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
        with self.audio_lock:
            self.audio_buf.extend(samples.tolist())

    def get_audio_window(self):
        """Return a float32 numpy array of the last 3 seconds, or None."""
        with self.audio_lock:
            if len(self.audio_buf) >= AUDIO_WINDOW_SAMPLES:
                return np.array(list(self.audio_buf)[-AUDIO_WINDOW_SAMPLES:],
                                dtype=np.float32)
        return None

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
        "piezo":         int(1800 + 600 * math.sin(2 * math.pi * 2.0 * t)
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
    while True:
        store.update(generate_simulated_packet(seq))
        seq += 1
        time.sleep(0.05)


# ── BLE mode ──────────────────────────────────────────────────────────────────

async def ble_loop():
    print("Scanning for ESP32 BLE device...")
    device = None

    while device is None:
        found = await BleakScanner.discover(timeout=5.0)
        for d in found:
            if d.name == BLE_DEVICE_NAME:
                device = d
                break
        if device is None:
            print("ESP32 not found. Retrying...")

    print(f"Found: {device.name} [{device.address}]")

    async with BleakClient(device.address) as client:
        # Request larger MTU to match the ESP32's 517-byte MTU setting
        try:
            await client.request_mtu(512)
        except Exception:
            pass  # not all platforms support explicit MTU negotiation

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
        await client.start_notify(AUDIO_CHAR_UUID, audio_handler)

        print("Subscribed to vitals and audio characteristics.")

        while True:
            await asyncio.sleep(1)


def start_data_receiver():
    if USE_BLE:
        thread = Thread(target=lambda: asyncio.run(ble_loop()), daemon=True)
    else:
        thread = Thread(target=simulation_loop, daemon=True)
    thread.start()
