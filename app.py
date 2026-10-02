"""
A818_NF — Brain Signal Decoder
Production BCI Software · v4.0 (Cloud-Ready)
"""

import streamlit as st
import numpy as np
import joblib
import os
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import butter, filtfilt, iirnotch

st.set_page_config(page_title="A818_NF — Brain Signal Decoder",
                   page_icon="🧠", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
.main {background-color:#0a0d12;}
h1 {color:#00d4ff; font-weight:700; letter-spacing:-0.5px;}
h2,h3 {color:#e6e9ef; font-weight:600;}
section[data-testid="stSidebar"] {background-color:#0f1419;}
.stButton>button {background-color:#00d4ff; color:#0a0d12; font-weight:700;
  border-radius:6px; padding:8px 18px; border:none; font-size:14px; transition:0.2s;}
.stButton>button:hover {background-color:#00b8e6; transform:translateY(-1px);}
.stButton>button:disabled {background-color:#2a2f3a !important; color:#666 !important;
  cursor:not-allowed !important;}
.status-bar {background:linear-gradient(90deg,#1a1f26,#0f1419); padding:12px 20px;
  border-radius:8px; border-left:4px solid #00d4ff; margin-bottom:20px; font-size:13px;}
.status-live{color:#00ff88; font-weight:700;}
.status-idle{color:#888; font-weight:600;}
.status-cal{color:#ffaa00; font-weight:700;}
.status-nowear{color:#ff4757; font-weight:700;}
.pred-card{background:linear-gradient(135deg,#1c1f26,#151920); padding:28px;
  border-radius:12px; border-left:5px solid #00d4ff; box-shadow:0 4px 20px rgba(0,212,255,0.08);}
.pred-card-5{border-left-color:#ff9f1c;}
.pred-label{color:#6b7280; font-size:11px; letter-spacing:2px; text-transform:uppercase; margin:0;}
.pred-value{font-size:34px; font-weight:800; margin:6px 0 0 0; letter-spacing:-1px;}
.true-label{color:#9ca3af; font-size:14px; margin:16px 0 4px 0;}
.true-value{color:#e6e9ef; font-size:20px; font-weight:600; margin:0;}
.correct{color:#00ff88; font-weight:700; font-size:14px; margin-top:14px;}
.incorrect{color:#ff4757; font-weight:700; font-size:14px; margin-top:14px;}
.conf-row{display:flex; align-items:center; margin-bottom:10px;}
.conf-name{color:#9ca3af; font-size:13px; width:120px; font-weight:500;}
.conf-bar-bg{flex:1; height:8px; background:#1a1f26; border-radius:4px; overflow:hidden;}
.conf-bar-fill{height:100%; background:linear-gradient(90deg,#00d4ff,#00b8e6); border-radius:4px;}
.conf-bar-fill-5{background:linear-gradient(90deg,#ff9f1c,#ff7b00);}
.conf-pct{color:#e6e9ef; font-size:12px; margin-left:12px; width:60px;
  text-align:right; font-variant-numeric:tabular-nums;}
.stat-box{background:#131720; padding:16px 12px; border-radius:8px; text-align:center;
  border:1px solid #1f242e;}
.stat-value{color:#00d4ff; font-size:22px; font-weight:700; font-variant-numeric:tabular-nums;}
.stat-label{color:#6b7280; font-size:10px; text-transform:uppercase; letter-spacing:1.5px; margin-top:4px;}
.mode-badge{display:inline-block; padding:6px 14px; border-radius:20px; font-size:12px;
  font-weight:700; letter-spacing:1px; margin-bottom:16px;}
.badge-3{background:rgba(0,212,255,0.15); color:#00d4ff; border:1px solid #00d4ff;}
.badge-5{background:rgba(255,159,28,0.15); color:#ff9f1c; border:1px solid #ff9f1c;}
.footer{text-align:center; color:#4b5563; font-size:11px; padding:30px 0 10px 0; letter-spacing:1px;}
.log-entry{padding:6px 12px; border-left:3px solid #2a2f3a; margin-bottom:4px;
  font-size:12px; color:#9ca3af; font-family:'Courier New',monospace;}
.log-entry-match{border-left-color:#00ff88;}
.log-entry-miss{border-left-color:#ff4757;}
.device-banner{background:#1a1f26; padding:18px 22px; border-radius:10px;
  border-left:5px solid #ff4757; margin:16px 0; color:#e6e9ef;}
.device-banner h3 {margin:0 0 8px 0; color:#ff4757; font-size:16px;}
.device-banner p {margin:4px 0; font-size:13px; color:#9ca3af;}
.device-banner code {background:#0a0d12; padding:2px 8px; border-radius:4px;
  color:#00d4ff; font-size:12px;}
.device-ok{background:#0f1a14; border-left-color:#00ff88;}
.device-ok h3 {color:#00ff88;}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# DEVICE INTERFACE
# ═══════════════════════════════════════════════════════════
class EEGDeviceInterface:
    @staticmethod
    def detect():
        return (False, None, "No EEG headband detected. Connect via Bluetooth.")

    @staticmethod
    def read_sample(duration_sec=3.0, fs=160, n_channels=64):
        return None

def get_device_status():
    return EEGDeviceInterface.detect()

# ═══════════════════════════════════════════════════════════
# PREPROCESSING
# ═══════════════════════════════════════════════════════════
def preprocess_eegnet(sample, fs=160):
    sample = sample.astype(np.float32)
    b_n, a_n = iirnotch(50, Q=30, fs=fs)
    x = filtfilt(b_n, a_n, sample, axis=-1)
    b, a = butter(4, [8/(fs/2), 30/(fs/2)], btype='band')
    x = filtfilt(b, a, x, axis=-1)
    x = (x - x.mean(axis=-1, keepdims=True)) / (x.std(axis=-1, keepdims=True) + 1e-8)
    return x.astype(np.float32)

# ═══════════════════════════════════════════════════════════
# EEGNET
# ═══════════════════════════════════════════════════════════
class EEGNet(nn.Module):
    def __init__(self, n_classes=5, n_channels=64, n_time=481):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, (1, 32), padding=(0, 16), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (n_channels, 1), groups=16, bias=False)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(32, 32, (1, 16), padding=(0, 8), groups=32, bias=False)
        self.conv4 = nn.Conv2d(32, 64, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool2 = nn.AvgPool2d((1, 4))
        self.drop2 = nn.Dropout(0.5)
        self.fc = nn.Linear(64 * (n_time // 16), n_classes)
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.bn1(self.conv1(x))
        x = self.bn2(self.conv2(x))
        x = torch.relu(x)
        x = self.drop1(self.pool1(x))
        x = self.conv3(x)
        x = self.bn3(self.conv4(x))
        x = torch.relu(x)
        x = self.drop2(self.pool2(x))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ═══════════════════════════════════════════════════════════
# PATHS — cloud-first, local files
# ═══════════════════════════════════════════════════════════
APP_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
DRIVE = '/content/drive/MyDrive/NeuroFlow'

def find_path(filename):
    # Priority: app directory (GitHub) → current dir → Drive (Colab)
    for base in [APP_DIR, os.getcwd(), DRIVE]:
        p = os.path.join(base, filename)
        if os.path.exists(p):
            return p
    return None

# Aliases so app works with either fp16 slim or original
ALIASES = {
    'app_data_X3.npy': ['app_data_X3.npy', 'app_data_X3_fp16.npy'],
    'app_data_X5.npy': ['app_data_X5.npy', 'app_data_X5_fp16.npy'],
}

def find_any(primary):
    names = ALIASES.get(primary, [primary])
    for n in names:
        p = find_path(n)
        if p: return p
    return None

@st.cache_resource
def load_3class():
    m = find_path('best_model.pkl')
    s = find_path('scaler.pkl')
    if m and s:
        return joblib.load(m), joblib.load(s)
    return None, None

@st.cache_resource
def load_5class():
    p = find_path('A818_NF_5class_eegnet_v2.pth')
    if p:
        m = EEGNet()
        m.load_state_dict(torch.load(p, map_location='cpu'))
        m.eval()
        return m
    return None

@st.cache_data
def load_data_3class():
    X = find_any('app_data_X3.npy')
    y = find_path('app_data_y3.npy')
    if X and y:
        return np.load(X).astype(np.float32), np.load(y)
    # Fallback to original big file
    X = find_path('X_all.npy'); y = find_path('y_all.npy')
    if X and y:
        return np.load(X), np.load(y)
    return None, None

@st.cache_data
def load_data_5class():
    X = find_any('app_data_X5.npy')
    y = find_path('app_data_y5.npy')
    if X and y:
        return np.load(X).astype(np.float32), np.load(y)
    X = find_path('A818_NF_5class_X_bal.npy')
    y = find_path('A818_NF_5class_y_bal.npy')
    if X and y:
        return np.load(X), np.load(y)
    return None, None

# ═══════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════
def init_state():
    defaults = {
        'buffer': None, 'playing': False, 'current_sample': None,
        'current_true': None, 'prediction': None, 'confidence': None,
        'session_log': [], 'samples_processed': 0, 'calibrating': False,
        'calibration_data': [], 'last_mode_key': None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_state()

# ═══════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════
st.title("🧠 A818_NF — Brain Signal Decoder")
st.markdown("##### Production BCI Software · Real-time intent classification")

# ═══════════════════════════════════════════════════════════
# DEVICE
# ═══════════════════════════════════════════════════════════
connected, device_name, device_msg = get_device_status()

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════
st.sidebar.markdown("## ⚙️ Control Panel")
st.sidebar.markdown("---")

mode = st.sidebar.radio("**Decoder Mode**", ["🎯 Focus Mode (3-class)", "🔬 Extended Mode (5-class)"])
operating_mode = st.sidebar.radio("**Operating Mode**", ["🧪 Bench Test", "🧬 Live Recording"])
live_mode = "Live" in operating_mode

mode_key = f"{mode}|{operating_mode}"
if st.session_state.last_mode_key != mode_key:
    st.session_state.buffer = None
    st.session_state.current_sample = None
    st.session_state.current_true = None
    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.session_log = []
    st.session_state.samples_processed = 0
    st.session_state.playing = False
    st.session_state.last_mode_key = mode_key

st.sidebar.markdown("---")

if "Focus" in mode:
    st.sidebar.markdown("**Model:** Logistic Regression")
    st.sidebar.markdown("**Validated accuracy:** 91.0%")
    st.sidebar.markdown("**Classes:** Rest · Left Fist · Right Fist")
else:
    st.sidebar.markdown("**Model:** EEGNet (CNN)")
    st.sidebar.markdown("**Validated accuracy:** 78.4%")
    st.sidebar.markdown("**Classes:** Rest · L.Fist · R.Fist · Both Fists · Both Feet")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔌 Device")
if connected:
    st.sidebar.success(f"🟢 {device_name}")
else:
    st.sidebar.error("🔴 No headband")
    st.sidebar.caption(device_msg)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Actions")

if "Focus" in mode:
    model, scaler = load_3class()
    X_data, y_data = load_data_3class()
    label_map = {1:"Rest", 2:"Left Fist", 3:"Right Fist"}
    accent = "#00d4ff"; badge_class = "badge-3"
    badge_text = "FOCUS MODE · 3 CLASSES · 91%"
else:
    model = load_5class(); scaler = None
    X_data, y_data = load_data_5class()
    label_map = {0:"Rest",1:"Left Fist",2:"Right Fist",3:"Both Fists",4:"Both Feet"}
    accent = "#ff9f1c"; badge_class = "badge-5"
    badge_text = "EXTENDED MODE · 5 CLASSES · 78%"

if model is None or X_data is None:
    st.error("⚠️ Model or data files not found. Check repository contents.")
    st.stop()

block_live = live_mode and not connected
if block_live:
    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.current_sample = None
    st.session_state.buffer = None
    st.session_state.playing = False

col_a, col_b = st.sidebar.columns(2)
with col_a: start_btn = st.button("▶ Stream", use_container_width=True, disabled=block_live)
with col_b: stop_btn = st.button("⏹ Stop", use_container_width=True)

col_c, col_d = st.sidebar.columns(2)
with col_c: single_btn = st.button("🎲 Sample", use_container_width=True, disabled=block_live)
with col_d: clear_btn = st.button("🧹 Clear", use_container_width=True)

if stop_btn: st.session_state.playing = False
if clear_btn:
    for k in ['buffer','current_sample','prediction','confidence']:
        st.session_state[k] = None
    st.session_state.session_log = []
    st.session_state.samples_processed = 0
    st.session_state.playing = False

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎓 Calibration")
calib_btn = st.sidebar.button("🧬 Start Calibration", use_container_width=True, disabled=block_live)
if calib_btn:
    st.session_state.calibrating = True
    st.session_state.calibration_data = []

# ═══════════════════════════════════════════════════════════
# BANNERS
# ═══════════════════════════════════════════════════════════
if live_mode and not connected:
    st.markdown("""
    <div class="device-banner">
        <h3>⚠️ No EEG Headband Detected</h3>
        <p><b>Live Recording requires an EEG headband.</b></p>
        <p>1. Wear the A818_NF headband</p>
        <p>2. Ensure electrodes are on scalp (forehead + earlobes)</p>
        <p>3. Turn on headband — LED should blink</p>
        <p>4. Pair via Bluetooth (Windows Settings → Bluetooth)</p>
        <p style="margin-top:14px;">Or switch to <code>🧪 Bench Test</code> to use saved dataset.</p>
    </div>
    """, unsafe_allow_html=True)
if live_mode and connected:
    st.markdown(f"""
    <div class="device-banner device-ok">
        <h3>🟢 {device_name} Connected</h3>
        <p>Streaming at 160 Hz · 64 channels · Ready to record.</p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# PREDICT
# ═══════════════════════════════════════════════════════════
def predict(sample):
    if "Focus" in mode:
        flat = sample.reshape(1, -1).astype(np.float32)
        fs = scaler.transform(flat)
        pred = int(model.predict(fs)[0])
        proba = model.predict_proba(fs)[0]
        return pred, proba
    else:
        processed = preprocess_eegnet(sample)
        with torch.no_grad():
            x = torch.tensor(processed, dtype=torch.float32).unsqueeze(0)
            logits = model(x)
            proba = torch.softmax(logits, dim=1).numpy()[0]
            return int(np.argmax(proba)), proba

# ═══════════════════════════════════════════════════════════
# ACTIONS
# ═══════════════════════════════════════════════════════════
import random

if not live_mode and X_data is not None:
    if single_btn:
        idx = random.randint(0, len(X_data)-1)
        sample = X_data[idx].astype(np.float32)
        pred, proba = predict(sample)
        st.session_state.current_sample = sample
        st.session_state.current_true = int(y_data[idx])
        st.session_state.prediction = pred
        st.session_state.confidence = proba
        st.session_state.samples_processed += 1
        st.session_state.session_log.append({
            'time': datetime.now().strftime('%H:%M:%S'),
            'pred': label_map[pred], 'true': label_map[int(y_data[idx])],
            'correct': pred == int(y_data[idx]), 'conf': float(max(proba))
        })

    if start_btn:
        st.session_state.playing = True
        if st.session_state.buffer is None:
            st.session_state.buffer = X_data[random.randint(0, len(X_data)-1)].astype(np.float32).copy()

    if st.session_state.playing:
        idx = random.randint(0, len(X_data)-1)
        new_sample = X_data[idx].astype(np.float32)
        if st.session_state.buffer is None:
            st.session_state.buffer = new_sample.copy()
        else:
            buf = st.session_state.buffer
            shift = 30
            buf = np.roll(buf, -shift, axis=-1)
            new_chunk = new_sample[:, -shift:] + np.random.randn(buf.shape[0], shift)*5
            buf[:, -shift:] = new_chunk
            st.session_state.buffer = buf

        if st.session_state.samples_processed % 3 == 0:
            pred, proba = predict(new_sample)
            st.session_state.prediction = pred
            st.session_state.confidence = proba
            st.session_state.current_true = int(y_data[idx])
            st.session_state.current_sample = new_sample
            st.session_state.session_log.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'pred': label_map[pred], 'true': label_map[int(y_data[idx])],
                'correct': pred == int(y_data[idx]), 'conf': float(max(proba))
            })
        st.session_state.samples_processed += 1

if live_mode and connected:
    if single_btn:
        sample = EEGDeviceInterface.read_sample()
        if sample is not None:
            pred, proba = predict(sample)
            st.session_state.current_sample = sample
            st.session_state.prediction = pred
            st.session_state.confidence = proba
            st.session_state.current_true = None
            st.session_state.samples_processed += 1
            st.session_state.session_log.append({
                'time': datetime.now().strftime('%H:%M:%S'),
                'pred': label_map[pred], 'true': None,
                'correct': None, 'conf': float(max(proba))
            })

    if start_btn: st.session_state.playing = True

    if st.session_state.playing:
        sample = EEGDeviceInterface.read_sample()
        if sample is not None:
            if st.session_state.buffer is None:
                st.session_state.buffer = sample
            else:
                st.session_state.buffer = np.concatenate(
                    [st.session_state.buffer[:, 30:], sample[:, -30:]], axis=-1)
            if st.session_state.samples_processed % 3 == 0:
                pred, proba = predict(sample)
                st.session_state.prediction = pred
                st.session_state.confidence = proba
                st.session_state.current_sample = sample
                st.session_state.session_log.append({
                    'time': datetime.now().strftime('%H:%M:%S'),
                    'pred': label_map[pred], 'true': None,
                    'correct': None, 'conf': float(max(proba))
                })
            st.session_state.samples_processed += 1

if len(st.session_state.session_log) > 20:
    st.session_state.session_log = st.session_state.session_log[-20:]

# ═══════════════════════════════════════════════════════════
# STATUS BAR
# ═══════════════════════════════════════════════════════════
s = '<div class="status-bar">'
if block_live: s += '<span class="status-nowear">🔴 NO DEVICE — WEAR HEADBAND</span>'
elif st.session_state.playing: s += '<span class="status-live">🟢 STREAMING</span>'
elif st.session_state.calibrating: s += '<span class="status-cal">🟡 CALIBRATING</span>'
else: s += '<span class="status-idle">🔴 IDLE</span>'
s += f' &nbsp;|&nbsp; Mode: <b>{operating_mode}</b>'
s += f' &nbsp;|&nbsp; Decoder: <b>{mode}</b>'
s += f' &nbsp;|&nbsp; Device: <b>{"Connected" if connected else "None"}</b>'
s += f' &nbsp;|&nbsp; Samples: <b>{st.session_state.samples_processed}</b>'
s += f' &nbsp;|&nbsp; Time: <b>{datetime.now().strftime("%H:%M:%S")}</b>'
s += '</div>'
st.markdown(s, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
# CALIBRATION
# ═══════════════════════════════════════════════════════════
if st.session_state.calibrating:
    st.markdown("---")
    st.markdown("## 🎓 Calibration Session")
    if live_mode and not connected:
        st.error("⚠️ Headband not detected. Wear and adjust the headband before recording.")
        st.markdown("""
        **Checklist:**
        - Headband on head
        - Electrodes touching scalp
        - Battery charged
        - Bluetooth paired
        - LED blinking
        """)
        if st.button("❌ Cancel Calibration"):
            st.session_state.calibrating = False
            st.rerun()
    else:
        st.info(f"**Collected:** {len(st.session_state.calibration_data)} samples. "
                f"Hold each thought for ~4 seconds while recording.")
        classes = ['Rest','Left Fist','Right Fist'] if "Focus" in mode else \
                  ['Rest','Left Fist','Right Fist','Both Fists','Both Feet']
        for cls_name in classes:
            c1, c2 = st.columns([3,1])
            with c1: st.markdown(f"**{cls_name}**")
            with c2:
                if st.button(f"● Record", key=f"cal_{cls_name}"):
                    if live_mode:
                        sample = EEGDeviceInterface.read_sample()
                        if sample is None:
                            st.error("Failed to read from device.")
                        else:
                            st.session_state.calibration_data.append((sample, cls_name))
                            st.success(f"Recorded: {cls_name}")
                    else:
                        idx = random.randint(0, len(X_data)-1)
                        st.session_state.calibration_data.append((X_data[idx].copy(), cls_name))
                        st.success(f"Bench sample: {cls_name}")
                    st.rerun()
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Finish Calibration", type="primary"):
                n = len(st.session_state.calibration_data)
                if n < 5: st.warning(f"Only {n} samples.")
                else:
                    st.session_state.calibrating = False
                    st.success(f"Calibration complete · {n} samples.")
        with c2:
            if st.button("❌ Cancel"):
                st.session_state.calibrating = False
                st.session_state.calibration_data = []
                st.rerun()
        st.markdown("---")

# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════
st.markdown(f'<div class="mode-badge {badge_class}">{badge_text}</div>', unsafe_allow_html=True)
col_left, col_right = st.columns([1,1])

with col_left:
    st.markdown("### 📊 Prediction")
    if st.session_state.prediction is not None and not block_live:
        pred = st.session_state.prediction
        proba = st.session_state.confidence
        pred_name = label_map[pred]
        card_class = "pred-card" if "Focus" in mode else "pred-card pred-card-5"
        pred_color = "#00d4ff" if "Focus" in mode else "#ff9f1c"
        h = f'<div class="{card_class}">'
        h += f'<p class="pred-label">Predicted Intent</p>'
        h += f'<p class="pred-value" style="color:{pred_color};">{pred_name}</p>'
        if not live_mode and st.session_state.current_true is not None:
            true_name = label_map[st.session_state.current_true]
            h += f'<p class="true-label">Ground Truth (Bench Test)</p>'
            h += f'<p class="true-value">{true_name}</p>'
            h += '<p class="correct">✅ Correct</p>' if pred == st.session_state.current_true else '<p class="incorrect">❌ Incorrect</p>'
        else:
            h += f'<p class="true-label">Mode</p>'
            h += f'<p class="true-value">Live — no ground truth</p>'
        h += '</div>'
        st.markdown(h, unsafe_allow_html=True)
    else:
        if block_live:
            st.warning("🔴 Waiting for headband. Wear the A818_NF headband to begin.")
        else:
            st.info("Click **▶ Stream** or **🎲 Sample** to begin.")

with col_right:
    st.markdown("### 📈 Confidence Distribution")
    if st.session_state.confidence is not None and not block_live:
        proba = st.session_state.confidence
        bar_class = "conf-bar-fill" if "Focus" in mode else "conf-bar-fill conf-bar-fill-5"
        for i, cls in label_map.items():
            if i < len(proba):
                pct = float(proba[i])*100
                st.markdown(f'<div class="conf-row"><div class="conf-name">{cls}</div>'
                            f'<div class="conf-bar-bg"><div class="{bar_class}" style="width:{pct}%;"></div></div>'
                            f'<div class="conf-pct">{pct:.2f}%</div></div>',
                            unsafe_allow_html=True)
    else:
        st.info("Awaiting signal...")

# ═══════════════════════════════════════════════════════════
# GRAPH
# ═══════════════════════════════════════════════════════════
st.markdown("### 🧠 Live EEG Waveforms")
buf = None
if not block_live:
    buf = st.session_state.buffer if st.session_state.buffer is not None else st.session_state.current_sample

if buf is not None and not block_live:
    n_ch = min(6, buf.shape[0])
    fig, axes = plt.subplots(n_ch, 1, figsize=(13,6), sharex=True)
    fig.patch.set_facecolor('#0a0d12')
    if n_ch == 1: axes = [axes]
    for i, ax in enumerate(axes):
        ax.set_facecolor('#0a0d12')
        ax.plot(buf[i], linewidth=0.7, color=accent)
        ax.set_ylabel(f'Ch{i+1}', color='#9ca3af', fontsize=8, rotation=0, labelpad=20)
        ax.tick_params(colors='#9ca3af', labelsize=8)
        ax.grid(True, alpha=0.12, color='#2a2f3a')
        for spine in ax.spines.values(): spine.set_color('#2a2f3a')
    axes[-1].set_xlabel('Time (samples)', color='#9ca3af', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    st.markdown("### 📉 Signal Metrics")
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.markdown(f'<div class="stat-box"><div class="stat-value">{buf.min()*1000:.2f}</div><div class="stat-label">Min (mV)</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="stat-box"><div class="stat-value">{buf.max()*1000:.2f}</div><div class="stat-label">Max (mV)</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="stat-box"><div class="stat-value">{buf.std()*1000:.2f}</div><div class="stat-label">Std (mV)</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="stat-box"><div class="stat-value">{buf.shape[0]}</div><div class="stat-label">Channels</div></div>', unsafe_allow_html=True)
else:
    if block_live: st.info("🔴 No signal. Wear the headband to start streaming.")
    else: st.info("No signal buffer yet.")

# ═══════════════════════════════════════════════════════════
# LOG
# ═══════════════════════════════════════════════════════════
st.markdown("### 📋 Session Log")
if st.session_state.session_log and not block_live:
    for e in reversed(st.session_state.session_log[-10:]):
        if e['true'] is not None:
            cls = "log-entry-match" if e['correct'] else "log-entry-miss"
            ic = "✅" if e['correct'] else "❌"
            line = f"[{e['time']}] {ic} Pred: {e['pred']} | True: {e['true']} | Conf: {e['conf']*100:.1f}%"
        else:
            cls = "log-entry"
            line = f"[{e['time']}] · Pred: {e['pred']} | Conf: {e['conf']*100:.1f}%"
        st.markdown(f'<div class="log-entry {cls}">{line}</div>', unsafe_allow_html=True)
else:
    st.info("Log empty." if block_live else "No predictions yet.")

st.markdown("---")
st.markdown('<div class="footer">A818_NF · Brain Signal Decoder · v4.0 · Built by Asma Rizwan Rao</div>', unsafe_allow_html=True)

if st.session_state.playing and not block_live:
    time.sleep(0.4)
    st.rerun()