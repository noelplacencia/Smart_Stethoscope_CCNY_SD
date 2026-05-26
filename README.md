# Smart Stethoscope

Wireless AI-assisted stethoscope for real-time respiratory and cardiac anomaly detection. Sensor data is processed at the edge on an ESP32, transmitted over BLE to a Raspberry Pi 4, and classified using a Random Forest model trained on the ICBHI 2017 and CirCor datasets.

Built for EE 59866/59868 Senior Design — The City College of New York.

---

## How it works

```
Sensors (MEMS mic, Piezo, ECG, SpO₂, IMU)
    → ESP32 (filtering, DSP, feature extraction)
        → BLE
            → Raspberry Pi 4 (Random Forest inference)
                → Dashboard (waveforms, anomaly flags, patient records)
```

---

## Repository structure

```
smart-stethoscope/
├── firmware/                # ESP32 Arduino code
│   └── main/
│       └── main.ino         # Sensor collection, DSP, BLE transmission
├── rpi/                     # Raspberry Pi Python code
│   ├── inference.py         # BLE (Bluetooth) receiver + RF (Recieved Feature) model inference
│   └── dashboard.py         # Real-time display and patient logging
├── ml/                      # Model training pipeline
│   ├── data/                # Not committed — see Datasets section below
│   │   ├── rf_model_lung.joblib       # Trained lung RF model
│   │   ├── rf_model_heart.joblib      # Trained heart RF model
│   │   ├── rf_model_piezo.joblib      # Trained piezo RF model
│   │   ├── scaler_lung.joblib         # StandardScaler for lung features
│   │   ├── scaler_heart.joblib        # StandardScaler for heart features
│   │   └── scaler_piezo.joblib        # StandardScaler for piezo features
│   ├── notebooks/           # Exploratory analysis
│   ├── extract_features_lung.py   # 38-feature extraction — ICBHI lung sounds (MEMS mic)
│   ├── extract_features_heart.py  # 17-feature extraction — CirCor heart sounds
│   ├── extract_features_piezo.py  # 38-feature extraction — ICBHI lung sounds (piezo sensor)
│   ├── train_lung.py              # Train and evaluate lung RF classifier
│   ├── train_heart.py             # Train and evaluate heart RF classifier
│   └── train_piezo.py             # Train and evaluate piezo RF classifier
├── docs/                    # Diagrams, reports, presentations
└── README.md
```

---

## Setup

### ESP32 (firmware)

1. Install [Arduino IDE](https://www.arduino.cc/en/software) and add ESP32 board support
2. Install required libraries via Arduino Library Manager:
   - `NimBLE-Arduino`
   - `Adafruit MAX3010x`
   - `MPU6050`
3. Open `firmware/main/main.ino` and upload to the ESP32

### Raspberry Pi (inference)

```bash
pip install bleak scikit-learn numpy joblib
python rpi/inference.py
```

Update `ESP32_ADDRESS` in `inference.py` with your device's BLE MAC address (find it by running `python -m bleak scan`).

### ML training (laptop)

```bash
pip install librosa scipy scikit-learn imbalanced-learn numpy pandas matplotlib joblib

# Lung sound model — MEMS mic (ICBHI 2017, 38 features)
python ml/extract_features_lung.py   # → ml/data/features_lung.csv
python ml/train_lung.py              # → ml/data/rf_model_lung.joblib, scaler_lung.joblib

# Heart sound model — murmur detection (CirCor DigiScope, 17 features)
python ml/extract_features_heart.py  # → ml/data/features_heart.csv
python ml/train_heart.py             # → ml/data/rf_model_heart.joblib, scaler_heart.joblib

# Piezo sensor model — chest wall vibration (ICBHI 2017, 38 features)
python ml/extract_features_piezo.py  # → ml/data/features_piezo.csv
python ml/train_piezo.py             # → ml/data/rf_model_piezo.joblib, scaler_piezo.joblib
```

Copy the `.joblib` files to the Pi before running inference.

---

## Datasets

Datasets are **not committed** to this repo due to file size. Download them manually and place in `ml/data/`:

| Dataset | Use | Link |
|---------|-----|------|
| ICBHI 2017 | Lung sound classification (normal, wheeze, crackle) | [bhichallenge.med.auth.gr](https://bhichallenge.med.auth.gr) |
| CirCor DigiScope | Heart murmur detection | [physionet.org/content/circor-heart-sound](https://physionet.org/content/circor-heart-sound/1.0.3/) |
| MIT-BIH | ECG arrhythmia classification | [physionet.org/content/mitdb](https://physionet.org/content/mitdb/1.0.0/) |

---

## Hardware

| Component | Role | Interface |
|-----------|------|-----------|
| SPH0645LM4H-B (MEMS mic) | Primary heart/lung audio | I²S |
| Ambient microphone | Noise reference for subtraction | I²S |
| Piezo sensor | Chest wall vibration | ADC |
| AD8232 | ECG / heart timing | ADC |
| MAX30102 | SpO₂ and pulse rate | I²C |
| MPU-6050 | Motion artifact detection | I²C |
| ESP32 | Edge DSP + BLE transmission | — |
| Raspberry Pi 4 | AI inference + dashboard | — |

---

## Contributing

We use a branch-per-feature workflow. Never commit directly to `main`.

```bash
# Start of every session
git checkout main
git pull origin main

# Create your branch
git checkout -b area/what-youre-doing
# e.g. ml/feature-extraction, firmware/ble-setup, rpi/dashboard-ui

# End of session
git add .
git commit -m "short description of what you did"
git push origin your-branch-name
```

Then open a Pull Request on GitHub to merge into `main`.

### Branch naming
- `ml/` — model training and feature extraction
- `firmware/` — ESP32 code
- `rpi/` — Raspberry Pi code
- `docs/` — reports, diagrams, slides

---

## Team

| Name | Role |
|------|------|
| Noel Placencia | Team Lead |
| Ulash Kundu Joy | Hardware Lead |
| Jason Corona | Filtering Lead|
| Saqlain Warrris | AI Lead|
| Puran Chaudharry | Data Lead |

---

## License

For academic use only. Not intended for clinical deployment.
