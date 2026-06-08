# Smart Stethoscope

Wireless AI-assisted stethoscope for real-time respiratory and cardiac anomaly detection. Audio is captured by a MEMS microphone, transmitted over BLE from an ESP32 to a Raspberry Pi 4, and classified using machine learning models trained on the ICBHI 2017, HF_Lung_V1, and CirCor DigiScope datasets.

Built for EE 59866/59868 Senior Design — The City College of New York.

---

## How it works

```
SPH0645 MEMS mic (heart/lung audio)
    → ESP32 (DSP filtering, BLE transmission)
        → Raspberry Pi 4 (ML inference)
            → Dashboard (waveforms, anomaly flags, patient records)
```

**Two inference modes:**
- **Lung mode** — classifies respiratory sounds as normal, crackle, wheeze, or both
- **Heart mode** — detects presence of cardiac murmur

---

## Repository structure

```
smart-stethoscope/
├── firmware/                        # Saqlain Warrris — DSP/filtering on ESP32
│   └── main/
│       └── main.ino
├── rpi/                             # Jason Corona — RPi inference engine
│   ├── ble_receiver.py
│   ├── app.py
│   ├── ai_alerts.py
│   ├── config.py
│   ├── requirements.txt
│   ├── run.sh
│   ├── static/
│   └── templates/                   # Puran Chaudharry — dashboard & patient logging
├── ml/                              # Noel Placencia — ML pipeline
│   ├── heart/
│   │   ├── extract_features.py      # CirCor feature extraction (53 features)
│   │   ├── train.py                 # RF+HGB Ensemble — murmur detection
│   │   └── train_cnn.py             # MobileNetV2 CNN — murmur detection
│   ├── lung/
│   │   ├── extract_features.py      # ICBHI feature extraction (38 features)
│   │   ├── extract_features_hf.py   # HF_Lung_V1 feature extraction
│   │   ├── extract_mels_hf.py       # HF_Lung_V1 mel spectrogram extraction
│   │   ├── train.py                 # Random Forest — lung sound classification
│   │   └── train_cnn.py             # MobileNetV2 CNN — lung sound classification
│   ├── piezo/
│   │   ├── extract_features.py
│   │   ├── train.py
│   │   └── train_cnn.py
│   ├── data/                        # not committed (models, caches, plots)
│   ├── datasets/                    # not committed (raw audio datasets)
│   └── show_metrics.py
├── docs/                            # Ulash Kundu Joy — hardware diagrams, reports
├── .gitignore
├── LICENSE
└── README.md
```

---

## Setup

### ESP32 (firmware)

1. Install [Arduino IDE](https://www.arduino.cc/en/software) and add ESP32 board support
2. Install required libraries via Arduino Library Manager:
   - `NimBLE-Arduino`
   - `Adafruit MAX3010x`
   - `BMI270`
3. Open `firmware/main/main.ino` and upload to the ESP32

### Raspberry Pi (inference)

```bash
pip install bleak scikit-learn numpy joblib
python rpi/inference.py
```

Update `ESP32_ADDRESS` in `inference.py` with your device's BLE MAC address (find it with `python -m bleak scan`).

### ML training

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install torch torchvision librosa scipy scikit-learn imbalanced-learn \
            numpy pandas matplotlib joblib
```

**Lung sound model** (MEMS mic — ICBHI 2017 + HF_Lung_V1, 4-class):
```bash
# Random Forest
python ml/lung/extract_features.py      # → ml/data/features_lung.csv
python ml/lung/extract_features_hf.py   # → ml/data/features_hf_lung.csv
python ml/lung/train.py                 # → ml/data/rf_model_lung.joblib

# CNN (MobileNetV2 transfer learning)
python ml/lung/extract_mels_hf.py       # → ml/data/mel_cache_hf.npz
python ml/lung/train_cnn.py             # → ml/data/cnn_model_lung.pth
```

**Heart sound model** (MEMS mic — CirCor DigiScope, binary murmur detection):
```bash
python ml/heart/extract_features.py    # → ml/data/features_heart.csv
python ml/heart/train.py               # → ml/data/rf_model_heart.joblib  (RF+HGB ensemble)

# CNN (MobileNetV2 transfer learning)
python ml/heart/train_cnn.py           # → ml/data/cnn_model_heart.pth
```

Copy the `.joblib` / `.pth` model files to the Pi before running inference.

---

## Model results

**Window-level** (each 3–5 second audio segment scored independently — primary training metric):

| Pipeline | Model | Dataset | Accuracy | ROC-AUC | Macro F1 |
|----------|-------|---------|----------|---------|----------|
| Heart | RF+HGB Ensemble | CirCor DigiScope | 79.5% | 0.696 | 0.634 |
| Heart | MobileNetV2 CNN | CirCor DigiScope | 70.0% | 0.765 | 0.620 |
| Lung | Random Forest | ICBHI + HF_Lung_V1 | 48.1% | 0.671 | — |
| Lung | MobileNetV2 CNN | ICBHI + HF_Lung_V1 | 43.1% | 0.680 | 0.350 |

**Patient-level** (all window probabilities for a patient are averaged into one score before classification):

| Pipeline | Model | Dataset | Accuracy | ROC-AUC | Macro F1 |
|----------|-------|---------|----------|---------|----------|
| Heart | RF+HGB Ensemble       | CirCor DigiScope   | 97.7% | 0.979 | 0.963 |
| Heart | MobileNetV2 CNN       | CirCor DigiScope   | 90.9% | 0.857 | 0.835 |
| Heart | RF+HGB + CNN Ensemble | CirCor DigiScope   | 97.1% | 0.971 | 0.954 |
| Lung  | Random Forest         | ICBHI + HF_Lung_V1 | 64.2% | —     | 0.374 |
| Lung  | MobileNetV2 CNN       | ICBHI + HF_Lung_V1 | 58.5% | —     | 0.471 |
| Lung  | RF + CNN Ensemble     | ICBHI + HF_Lung_V1 | 69.8% | —     | 0.516 |

> **Why are patient-level numbers so much higher?** Two reasons: (1) averaging 40–50 window predictions per patient cancels out per-window noise — a model that is only slightly better than random on each window will converge to the correct answer reliably when you average many windows together; (2) the patient-level test set is small (175 patients, 36 with murmur), so a few correct predictions swing AUC significantly. Window-level metrics are the honest measure of what the model learned. Patient-level metrics reflect how the system would actually be used in deployment (a full recording, not a single 5-second clip), but should be interpreted cautiously given the small patient count.

Lung classification is a 4-class problem (normal / crackle / wheeze / both) against a heavily imbalanced dataset — ROC-AUC is the primary metric.

---

## Datasets

Datasets are **not committed** to this repo due to file size. Download and place in `ml/datasets/`:

| Dataset | Use | Link |
|---------|-----|------|
| ICBHI 2017 | Lung sound classification | [bhichallenge.med.auth.gr](https://bhichallenge.med.auth.gr) |
| HF_Lung_V1 | Additional lung sounds (171 patients) | [Hugging Face](https://huggingface.co/datasets/Stethoscope/HF_Lung_V1) |
| CirCor DigiScope | Heart murmur detection | [physionet.org/content/circor-heart-sound](https://physionet.org/content/circor-heart-sound/1.0.3/) |

---

## Hardware

| Component | Role | Interface |
|-----------|------|-----------|
| SPH0645LM4H-B (MEMS mic) | Primary heart/lung audio capture | I²S |
| INMP441 (ambient mic) | Noise reference | I²S |
| Murata 7BB-27-4L0 (piezo ×2) | Chest wall vibration | ADC |
| AD8232 | ECG / heart timing | ADC |
| MAX30102 | SpO₂ and pulse rate | I²C |
| BMI270 | Motion artifact detection | I²C |
| DS18B20 | Body temperature | 1-Wire |
| SEN0297 | Chest pressure | ADC |
| ESP32 | Edge DSP + BLE transmission | — |
| Raspberry Pi 4 | ML inference + dashboard | — |

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
| Jason Corona | Filtering Lead |
| Saqlain Warrris | AI Lead |
| Puran Chaudharry | Data Lead |

---

## License

For academic use only. Not intended for clinical deployment.
