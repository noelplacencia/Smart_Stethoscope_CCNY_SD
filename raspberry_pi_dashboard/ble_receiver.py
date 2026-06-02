import asyncio
import json
import random
import math
import time
from threading import Thread, Lock
from collections import deque

from bleak import BleakClient, BleakScanner

from config import USE_BLE, BLE_DEVICE_NAME, CHARACTERISTIC_UUID, MAX_POINTS
from ai_alerts import analyze_vitals


class DataStore:
    def __init__(self):
        self.lock = Lock()
        self.latest = {}
        self.ecg_wave = deque(maxlen=MAX_POINTS)
        self.heart_wave = deque(maxlen=MAX_POINTS)
        self.spo2_trend = deque(maxlen=MAX_POINTS)
        self.resp_trend = deque(maxlen=MAX_POINTS)
        self.hr_trend = deque(maxlen=MAX_POINTS)
        self.time_axis = deque(maxlen=MAX_POINTS)

    def update(self, packet):
        with self.lock:
            packet["timestamp"] = time.strftime("%H:%M:%S")
            packet["ai_result"] = analyze_vitals(packet)

            self.latest = packet
            self.ecg_wave.append(packet.get("ecg", 0))
            self.heart_wave.append(packet.get("heart_sound", 0))
            self.spo2_trend.append(packet.get("spo2", 0))
            self.resp_trend.append(packet.get("respiration", 0))
            self.hr_trend.append(packet.get("heart_rate", 0))
            self.time_axis.append(packet["timestamp"])

    def get_dashboard_data(self):
        with self.lock:
            return {
                "latest": self.latest,
                "ecg_wave": list(self.ecg_wave),
                "heart_wave": list(self.heart_wave),
                "spo2_trend": list(self.spo2_trend),
                "resp_trend": list(self.resp_trend),
                "hr_trend": list(self.hr_trend),
                "time_axis": list(self.time_axis)
            }


store = DataStore()


def generate_simulated_packet(seq):
    t = time.time()
    ecg = int(2050 + 550 * math.sin(2 * math.pi * 1.2 * t))
    if seq % 20 == 0:
        ecg += 1200

    heart_sound = int(1800 + 600 * math.sin(2 * math.pi * 2.0 * t) + random.randint(-100, 100))

    return {
        "seq": seq,
        "ecg": ecg,
        "heart_sound": heart_sound,
        "ambient_sound": random.randint(1200, 1800),
        "heart_rate": random.randint(72, 82),
        "spo2": random.randint(96, 99),
        "respiration": random.randint(14, 18),
        "temperature": round(random.uniform(36.5, 37.0), 2),
        "pressure": random.randint(1500, 2500),
        "piezo1": random.randint(1500, 2500),
        "piezo2": random.randint(1500, 2500),
        "imu_x": round(random.uniform(-0.05, 0.05), 3),
        "imu_y": round(random.uniform(-0.05, 0.05), 3),
        "imu_z": round(random.uniform(0.95, 1.05), 3),
    }


def simulation_loop():
    seq = 0
    while True:
        packet = generate_simulated_packet(seq)
        store.update(packet)
        seq += 1
        time.sleep(0.05)


async def ble_loop():
    print("Scanning for ESP32 BLE device...")
    device = None

    while device is None:
        devices = await BleakScanner.discover(timeout=5.0)
        for d in devices:
            if d.name == BLE_DEVICE_NAME:
                device = d
                break

        if device is None:
            print("ESP32 not found. Retrying...")

    print(f"Found device: {device.name} [{device.address}]")

    async with BleakClient(device.address) as client:
        print("Connected to ESP32.")

        def notification_handler(sender, data):
            try:
                text = data.decode("utf-8")
                packet = json.loads(text)
                store.update(packet)
            except Exception as e:
                print("BLE packet error:", e)

        await client.start_notify(CHARACTERISTIC_UUID, notification_handler)

        while True:
            await asyncio.sleep(1)


def start_data_receiver():
    if USE_BLE:
        thread = Thread(target=lambda: asyncio.run(ble_loop()), daemon=True)
    else:
        thread = Thread(target=simulation_loop, daemon=True)

    thread.start()
