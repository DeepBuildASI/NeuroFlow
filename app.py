"""
A818_NF — Brain Signal Decoder
Professional BCI Interface
"""

import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(
    page_title="A818_NF — Brain Signal Decoder",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS
# ============================================
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    h1 { color: #00d4ff; font-family: 'Segoe UI', sans-serif; font-weight: 700; }
    h2, h3 { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; }
    .stButton>button {
        background-color: #00d4ff;
        color: #0e1117;
        font-weight: 700;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-size: 15px;
    }
    .stButton>button:hover { background-color: #00b8e6; }
    .mode-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .badge-3 { background-color: #00d4ff; color: #0e1117; }
    .badge-5 { background-color: #ff9f1c; color: #0e1117; }
    .prediction-card {
        background-color: #1c1f26;
        padding: 24px;
        border-radius: 12px;
        border-left: 5px solid #00d4ff;
    }
    .stat-box {
        background-color: #1c1f26;
        padding: 16px;
        border-radius: 8px;
        text-align: center;
        border: 1px solid #2a2f3a;
    }
    .stat-value { color: #00d4ff; font-size: 26px; font-weight: 700; }
    .stat-label { color: #888; font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
    .footer {
        text-align: center;
        color: #555;
        font-size: 12px;
        padding: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# EEGNET ARCHITECTURE
# ============================================
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

# ============================================
# LOAD MODELS
# ============================================
DRIVE = '/content/drive/MyDrive/NeuroFlow'
LOCAL = os.path.dirname(os.path.abspath(__file__))

def find_path(filename):
    """Try Drive first, then local"""
    drive_path = f'{DRIVE}/{filename}'
    local_path = os.path.join(LOCAL, filename)
    if os.path.exists(drive_path):
        return drive_path
    if os.path.exists(local_path):
        return local_path
    return None

@st.cache_resource
def load_3class():
    model_path = find_path('best_model.pkl')
    scaler_path = find_path('scaler.pkl')
    if model_path and scaler_path:
        return joblib.load(model_path), joblib.load(scaler_path)
    return None, None

@st.cache_resource
def load_5class():
    eegnet_path = find_path('A818_NF_5class_eegnet_v2.pth')
    if eegnet_path:
        model = EEGNet()
        model.load_state_dict(torch.load(eegnet_path, map_location='cpu'))
        model.eval()
        return model
    return None

@st.cache_data
def load_5class_data():
    X_path = find_path('A818_NF_5class_X_bal.npy')
    y_path = find_path('A818_NF_5class_y_bal.npy')
    if X_path and y_path:
        return np.load(X_path), np.load(y_path)
    return None, None

# ============================================
# HEADER
# ============================================
st.title("🧠 A818_NF — Brain Signal Decoder")
st.markdown("### Decode human intent from EEG signals using AI")

# ============================================
# SIDEBAR
# ============================================
st.sidebar.markdown("## ⚙️ Control Panel")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "**Select Mode:**",
    ["🎯 Focus Mode (3 classes)", "🔬 Extended Mode (5 classes)"]
)

st.sidebar.markdown("---")

if "Focus" in mode:
    st.sidebar.markdown("**Model:** Logistic Regression")
    st.sidebar.markdown("**Accuracy:** 91%")
    st.sidebar.markdown("**Classes:** 3")
    st.sidebar.markdown("**Training samples:** 2,820")
    st.sidebar.markdown("**Calibration:** Not required")
else:
    st.sidebar.markdown("**Model:** EEGNet (CNN)")
    st.sidebar.markdown("**Accuracy:** 78%")
    st.sidebar.markdown("**Classes:** 5")
    st.sidebar.markdown("**Training samples:** 4,500")
    st.sidebar.markdown("**Calibration:** 20 min")

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Actions")

if 'sample' not in st.session_state:
    st.session_state.sample = None
    st.session_state.true_label = None
    st.session_state.pred = None
    st.session_state.proba = None
    st.session_state.mode_used = None

col_a, col_b = st.sidebar.columns(2)

with col_a:
    predict_btn = st.button("🎲 Predict")
with col_b:
    clear_btn = st.button("🧹 Clear")

# ============================================
# MODE 1: 3-CLASS FOCUS
# ============================================
if "Focus" in mode:
    st.markdown('<div class="mode-badge badge-3">FOCUS MODE · 3 CLASSES · 91% ACCURACY</div>', unsafe_allow_html=True)

    model_3, scaler_3 = load_3class()

    if model_3 is None:
        st.error("3-class model not found. Make sure `best_model.pkl` and `scaler.pkl` are on Drive.")
        st.stop()

    X_path = find_path('X_all.npy')
    y_path = find_path('y_all.npy')

    if X_path and y_path:
        X_data = np.load(X_path)
        y_data = np.load(y_path)

        if predict_btn:
            idx = np.random.randint(0, len(X_data))
            st.session_state.sample = X_data[idx]
            st.session_state.true_label = y_data[idx]
            flat = X_data[idx].reshape(1, -1)
            flat_s = scaler_3.transform(flat)
            st.session_state.pred = model_3.predict(flat_s)[0]
            st.session_state.proba = model_3.predict_proba(flat_s)[0]
            st.session_state.mode_used = "3class"

        if clear_btn:
            st.session_state.sample = None

        if st.session_state.sample is not None and st.session_state.mode_used == "3class":
            label_map = {1: "Rest", 2: "Left Fist", 3: "Right Fist"}
            pred = st.session_state.pred
            true_label = st.session_state.true_label
            proba = st.session_state.proba
            sample = st.session_state.sample

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Prediction")
                st.markdown(f"""
                <div class="prediction-card">
                    <p style="color:#888; font-size:13px; margin:0;">PREDICTED</p>
                    <p style="color:#00d4ff; font-size:32px; font-weight:700; margin:5px 0;">{label_map[pred]}</p>
                    <p style="color:#888; font-size:13px; margin:15px 0 5px 0;">TRUE LABEL</p>
                    <p style="color:#e0e0e0; font-size:20px; margin:0;">{label_map[true_label]}</p>
                    <p style="margin:15px 0 0 0; font-size:14px; color:{'#00ff88' if pred == true_label else '#ff5555'};">
                        {'✅ CORRECT' if pred == true_label else '❌ INCORRECT'}
                    </p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown("### 📈 Confidence")
                for i, cls in enumerate([1, 2, 3]):
                    st.markdown(f"**{label_map[cls]}**: {proba[i]*100:.2f}%")
                    st.progress(float(proba[i]))

            st.markdown("### 🧠 EEG Signal")
            fig, ax = plt.subplots(figsize=(12, 3))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            for ch in range(min(5, sample.shape[0])):
                ax.plot(sample[ch] + ch*100, linewidth=0.7, color='#00d4ff')
            ax.set_xlabel('Time (samples)', color='white')
            ax.set_ylabel('Channel', color='white')
            ax.tick_params(colors='white')
            ax.grid(True, alpha=0.2)
            st.pyplot(fig)
        else:
            st.info("👈 Click **🎲 Predict** in the sidebar to decode a brain sample.")

# ============================================
# MODE 2: 5-CLASS EXTENDED
# ============================================
else:
    st.markdown('<div class="mode-badge badge-5">EXTENDED MODE · 5 CLASSES · 78% ACCURACY</div>', unsafe_allow_html=True)

    model_5 = load_5class()
    X_data, y_data = load_5class_data()

    if model_5 is None or X_data is None:
        st.error("5-class model not found. Make sure `A818_NF_5class_eegnet_v2.pth`, `A818_NF_5class_X_bal.npy`, and `A818_NF_5class_y_bal.npy` are on Drive.")
        st.stop()

    if predict_btn:
        idx = np.random.randint(0, len(X_data))
        sample = X_data[idx]
        st.session_state.sample = sample
        st.session_state.true_label = y_data[idx]

        with torch.no_grad():
            x_t = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
            logits = model_5(x_t)
            proba = torch.softmax(logits, dim=1).numpy()[0]
            pred = int(np.argmax(proba))

        st.session_state.pred = pred
        st.session_state.proba = proba
        st.session_state.mode_used = "5class"

    if clear_btn:
        st.session_state.sample = None

    if st.session_state.sample is not None and st.session_state.mode_used == "5class":
        label_map = {0: "Rest", 1: "Left Fist", 2: "Right Fist", 3: "Both Fists", 4: "Both Feet"}
        pred = st.session_state.pred
        true_label = st.session_state.true_label
        proba = st.session_state.proba
        sample = st.session_state.sample

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Prediction")
            st.markdown(f"""
            <div class="prediction-card">
                <p style="color:#888; font-size:13px; margin:0;">PREDICTED</p>
                <p style="color:#ff9f1c; font-size:32px; font-weight:700; margin:5px 0;">{label_map[pred]}</p>
                <p style="color:#888; font-size:13px; margin:15px 0 5px 0;">TRUE LABEL</p>
                <p style="color:#e0e0e0; font-size:20px; margin:0;">{label_map[true_label]}</p>
                <p style="margin:15px 0 0 0; font-size:14px; color:{'#00ff88' if pred == true_label else '#ff5555'};">
                    {'✅ CORRECT' if pred == true_label else '❌ INCORRECT'}
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("### 📈 Confidence")
            for i, cls in enumerate([0, 1, 2, 3, 4]):
                st.markdown(f"**{label_map[cls]}**: {proba[i]*100:.2f}%")
                st.progress(float(proba[i]))

        st.markdown("### 🧠 EEG Signal")
        fig, ax = plt.subplots(figsize=(12, 3))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#0e1117')
        for ch in range(min(5, sample.shape[0])):
            ax.plot(sample[ch] + ch*100, linewidth=0.7, color='#ff9f1c')
        ax.set_xlabel('Time (samples)', color='white')
        ax.set_ylabel('Channel', color='white')
        ax.tick_params(colors='white')
        ax.grid(True, alpha=0.2)
        st.pyplot(fig)

        st.markdown("### 📉 Signal Statistics")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{sample.min()*1e6:.1f}</div><div class="stat-label">Min (µV)</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{sample.max()*1e6:.1f}</div><div class="stat-label">Max (µV)</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="stat-box"><div class="stat-value">{sample.std()*1e6:.1f}</div><div class="stat-label">Std (µV)</div></div>', unsafe_allow_html=True)

    else:
        st.info("👈 Click **🎲 Predict** in the sidebar to decode a brain sample.")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown('<div class="footer">A818_NF · Brain Signal Decoder · v2.0<br>Built by Asma Rizwan Rao</div>', unsafe_allow_html=True)