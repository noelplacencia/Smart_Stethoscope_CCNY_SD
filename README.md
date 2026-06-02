# Smart Stethoscope Dashboard

Senior Design Smart Stethoscope software package.

Architecture:

```text
Sensors → ESP32-S3 → BLE → Raspberry Pi 4 → AI + Web Dashboard → Phone/Laptop/Tablet
```

This package includes:

- ESP32-S3 BLE firmware starter
- Raspberry Pi BLE receiver
- Flask web dashboard
- Doctor View
- Patient View
- AI alert module
- Simulated-data mode for testing without hardware
- GitHub-ready folder structure

## Raspberry Pi Installation

```bash
cd raspberry_pi_dashboard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open dashboard:

```text
http://raspberrypi.local:5000
```

or use the Pi IP address:

```text
http://192.168.x.x:5000
```

## Test Without ESP32

By default, the dashboard runs in simulation mode.

In `raspberry_pi_dashboard/config.py`:

```python
USE_BLE = False
```

To use ESP32 BLE:

```python
USE_BLE = True
```

## ESP32 Installation

1. Open `esp32_firmware/smart_stethoscope_ble.ino` in Arduino IDE.
2. Select board: ESP32S3 Dev Module.
3. Upload to ESP32-S3.
4. Run Raspberry Pi dashboard with `USE_BLE = True`.

## Your Contribution Statement

Developed the real-time Smart Stethoscope dashboard interface, including doctor view, patient view, BLE receiver integration, AI alert display, and multi-device web access.


## Post to GitHub

```bash
git init
git add .
git commit -m "Initial Smart Stethoscope dashboard code"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/smart-stethoscope-dashboard.git
git push -u origin main
```

## Demo Command

```bash
cd raspberry_pi_dashboard
source venv/bin/activate
python app.py
```

Then open:

```text
http://localhost:5000
```

For other devices on the same network, open the Raspberry Pi IP address:

```text
http://RASPBERRY_PI_IP:5000
```
