# NeuroFlow
AI-powered brain-computer interface
# 🧠 NeuroFlow

AI-powered brain signal decoder that predicts human intent from EEG data.

Built by Asma Rizwan Rao — 3rd semester student, LCWU.

## What it does
NeuroFlow reads raw brain signals (EEG) and predicts what a person is doing — Rest, Left Fist, or Right Fist — with 91% accuracy.

This is Phase 1 of a larger vision: a software-first brain-computer interface that works with a $200 headband — no surgery, no expensive implants.

## Model Performance
- Accuracy: 91%
- Training samples: 2,256
- Test samples: 564
- Classes: 3 (Rest, Left Fist, Right Fist)
- Channels: 64
- Data source: PhysioNet EEG Motor Movement

## How it works
Raw EEG (64 channels × 481 time points) → Preprocessing (MNE-Python) → Feature Extraction → AI Classification (Logistic Regression) → Prediction

## Files
- app.py — Streamlit web app
- NeuroFlow_v1.ipynb — Training notebook
- best_model.pkl — Trained classifier
- scaler.pkl — Feature normalizer
- data/ — Sample EEG files

## Roadmap
- Phase 1: 3-class decoder, 91% accuracy — DONE
- Phase 2: Expand to 6–10 classes
- Phase 3: Pair with EEG headband for live use
- Phase 4: Real-time mind-controlled computer demo
- Phase 5: Medical partnerships

## Tech Stack
Python, MNE-Python, scikit-learn, PyTorch, Streamlit, Google Colab

## Contact
GitHub: @deepbuildASI
LinkedIn: Asma Rizwan Rao

## License
MIT License — see LICENSE
