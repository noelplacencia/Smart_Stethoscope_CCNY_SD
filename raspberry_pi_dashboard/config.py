# Smart Stethoscope Configuration

# False = test dashboard without ESP32.
# True = receive real BLE packets from ESP32-S3.
USE_BLE = False

BLE_DEVICE_NAME = "SmartStethoscope_ESP32"

SERVICE_UUID = "7b4d0001-8a7b-4d2b-9b41-000000000001"
CHARACTERISTIC_UUID = "7b4d0002-8a7b-4d2b-9b41-000000000001"

HOST = "0.0.0.0"
PORT = 5000
MAX_POINTS = 250
