# 🧠 A818_NF — NeuroFlow

**AI-powered brain signal decoder · Real-time EEG intent classification**

Built by **Asma Rizwan Rao**

---

## Overview

A818_NF (NeuroFlow) is a production-ready brain-computer interface software that decodes human intent from raw EEG signals. Dual-mode architecture supports both a high-accuracy 3-class decoder and an extended 5-class decoder with per-subject calibration.

---

## Model Performance

| Mode | Classes | Accuracy | Model | Calibration |
|------|---------|----------|-------|-------------|
| **Focus Mode** | 3 (Rest · Left Fist · Right Fist) | **91.0%** | Logistic Regression | Not required |
| **Extended Mode** | 5 (Rest · L.Fist · R.Fist · Both Fists · Both Feet) | **78.4%** | EEGNet (CNN) | 20 min per subject |

- **3-class training:** 2,820 trials, 5 subjects
- **5-class training:** 4,500 trials (balanced), 20 subjects
- **Data source:** PhysioNet EEG Motor Movement/Imagery (109 subjects)
- **Channels:** 64 (3-class) / 22 (5-class, after harmonization)
- **Sampling rate:** 160 Hz

---

## Architecture
Raw EEG Signal
↓
Preprocessing (MNE-Python)
· Notch filter (50 Hz)
· Bandpass (8–30 Hz)
· Z-score normalization
↓
Feature Extraction / Deep Learning
· 3-class: Flatten + StandardScaler
· 5-class: EEGNet convolutional model
↓
Classification
· Focus Mode → Logistic Regression
· Extended Mode → EEGNet CNN
↓
Intent Prediction + Confidence Distribution

text

---

## Software Features

- **Dual-mode decoder** — switch between 3-class and 5-class
- **Two operating modes** — Bench Test (saved data) vs Live Recording (hardware)
- **Device-aware** — Live Mode refuses to run without EEG headband connected
- **Real-time EEG waveforms** — scrolling plot for 6 channels
- **Confidence distribution** — per-class probability bars
- **Session log** — timestamped predictions with correct/incorrect flags
- **Calibration workflow** — per-class recording for personalized models
- **Professional UI** — dark theme, live status indicators, no fake outputs

---

## Files

| File | Purpose |
|------|---------|
| `app.py` | Streamlit production app (dual-mode) |
| `A818_NF_PhysioNet_v2.ipynb` | 5-class training notebook |
| `NeuroFlow_v1.ipynb` | 3-class training notebook |
| `best_model.pkl` | 3-class Logistic Regression model |
| `scaler.pkl` | 3-class feature scaler |
| `A818_NF_5class_eegnet_v2.pth` | 5-class EEGNet weights |
| `A818_NF_5class_X_bal.npy` | Balanced 5-class training data |
| `A818_NF_5class_y_bal.npy` | 5-class labels |
| `data/` | Sample PhysioNet EEG files |

**Note:** Large data files (`.npy`, `.pth`) are hosted on Google Drive due to GitHub's 100 MB file size limit.

---

## Hardware Integration

The app includes a **device interface layer** for EEG headband integration. When hardware is connected:

1. Fill in `EEGDeviceInterface.detect()` — Bluetooth port scanner
2. Fill in `EEGDeviceInterface.read_sample()` — Bluetooth data reader
3. No other code changes needed

**Recommended hardware:** BioAmp EXG Pill + ESP32 (~$100)

---

## Roadmap

- ✅ **Phase 1** — 3-class decoder (91% accuracy)
- ✅ **Phase 2** — 5-class extended decoder (78% with calibration)
- ✅ **Phase 3** — Production dual-mode software
- 🔄 **Phase 4** — EEG headband hardware integration
- ⏳ **Phase 5** — Real-time live demo with users
- ⏳ **Phase 6** — Medical partnerships & clinical validation

---

## Tech Stack

**AI/ML:** PyTorch · scikit-learn · MNE-Python · NumPy · SciPy
**App:** Streamlit · Matplotlib
**Training:** Google Colab (T4 GPU)
**Deployment:** Cloudflare Tunnel (dev) · Streamlit Cloud (planned)

---

## Running Locally

```bash
pip install streamlit mne scikit-learn torch numpy matplotlib scipy
streamlit run app.py
Contact
GitHub: @deepbuildASI

LinkedIn: Asma Rizwan Rao

Email: asma.rizwanrao7@gmail.com

License
MIT License — see LICENSE