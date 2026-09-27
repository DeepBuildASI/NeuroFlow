import streamlit as st
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="NeuroFlow", page_icon="🧠", layout="wide")

st.markdown("""
<style>
h1 {color: #00d4ff;}
.stButton>button {background-color: #00d4ff; color: black; font-weight: bold; border-radius: 8px;}
</style>
""", unsafe_allow_html=True)

st.title("🧠 NeuroFlow — Brain Signal Decoder")
st.markdown("### Decode human intent from EEG signals using AI")

folder = '/content/drive/MyDrive/NeuroFlow'

@st.cache_data
def load_data():
    X = np.load(os.path.join(folder, 'X_all.npy'))
    y = np.load(os.path.join(folder, 'y_all.npy'))
    return X, y

@st.cache_resource
def load_model():
    scaler = joblib.load(os.path.join(folder, 'scaler.pkl'))
    model = joblib.load(os.path.join(folder, 'best_model.pkl'))
    return scaler, model

X, y = load_data()
scaler, model = load_model()

st.sidebar.header("⚙️ Model Info")
st.sidebar.markdown("**Model:** Logistic Regression")
st.sidebar.markdown("**Accuracy:** 91%")
st.sidebar.markdown("**Classes:** 3 (Rest, Left Fist, Right Fist)")
st.sidebar.markdown("**Training samples:** 2,256")

st.sidebar.markdown("---")
st.sidebar.header("🎛️ Actions")

if 'sample' not in st.session_state:
    st.session_state.sample = None
    st.session_state.true_label = None
    st.session_state.pred = None
    st.session_state.proba = None

col_btn1, col_btn2 = st.sidebar.columns(2)

with col_btn1:
    if st.button("🎲 Predict"):
        idx = np.random.randint(0, len(X))
        st.session_state.sample = X[idx]
        st.session_state.true_label = y[idx]
        flat = X[idx].reshape(1, -1)
        flat_scaled = scaler.transform(flat)
        st.session_state.pred = model.predict(flat_scaled)[0]
        st.session_state.proba = model.predict_proba(flat_scaled)[0]

with col_btn2:
    if st.button("🧹 Clear"):
        st.session_state.sample = None
        st.session_state.true_label = None
        st.session_state.pred = None
        st.session_state.proba = None

if st.session_state.sample is not None:
    label_map = {1: "Rest", 2: "Left Fist", 3: "Right Fist"}
    pred = st.session_state.pred
    true_label = st.session_state.true_label
    proba = st.session_state.proba
    sample = st.session_state.sample

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 📊 Prediction")
        st.success(f"**Predicted:** {label_map[pred]}")
        st.info(f"**True Label:** {label_map[true_label]}")
        if pred == true_label:
            st.markdown("✅ **Correct prediction**")
        else:
            st.markdown("❌ **Incorrect prediction**")

    with col2:
        st.markdown("### 📈 Confidence")
        for i, cls in enumerate([1, 2, 3]):
            st.markdown(f"**{label_map[cls]}**: {proba[i]*100:.2f}%")
            st.progress(float(proba[i]))

    st.markdown("### 🧠 Real EEG Signal (first 5 channels, raw microvolts)")
    fig, axes = plt.subplots(5, 1, figsize=(12, 8), sharex=True)
    for ch in range(5):
        axes[ch].plot(sample[ch], linewidth=0.6, color='#00d4ff')
        axes[ch].set_ylabel(f'Ch {ch+1}', fontsize=8)
        axes[ch].grid(True, alpha=0.2)
    axes[-1].set_xlabel('Time (samples)')
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### 📉 Signal Statistics")
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Min (μV)", f"{sample.min()*1e6:.2f}")
    col_b.metric("Max (μV)", f"{sample.max()*1e6:.2f}")
    col_c.metric("Std (μV)", f"{sample.std()*1e6:.2f}")   

else:
    st.info("👈 Click **🎲 Predict** in the sidebar to decode a random brain sample.")

st.markdown("---")
st.markdown("**NeuroFlow v1.0** | Built by Asma Rizwan Rao")
