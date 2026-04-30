import sys
sys.stdout.reconfigure(line_buffering=True)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import json
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="FL IDS — Smart Energy IoT",
    page_icon="⚡",
    layout="wide"
)

# ── Load model and artifacts ──────────────────────────────────────────
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("saved_models/best_model.h5")
    class_names = np.load(
        "saved_models/class_names.npy", allow_pickle=True)
    class_names = [str(c) for c in class_names]
    return model, class_names

@st.cache_data
def load_eval_results():
    try:
        with open("evaluation/results.json") as f:
            return json.load(f)
    except:
        return None

model, class_names = load_model()
num_classes = len(class_names)
WINDOW = 20

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/cyber-security.png", width=80)
    st.title("FL IDS System")
    st.caption("Federated Learning Intrusion Detection for Smart Energy IoT")
    st.divider()
    st.markdown("**Model:** CNN + BiLSTM + Attention")
    st.markdown("**Datasets:** CICIoT2023 + EdgeIIoTset")
    st.markdown(f"**Classes:** {', '.join(class_names)}")
    st.divider()
    page = st.radio("Navigate", [
        "🏠 Home",
        "🔍 Live Detection",
        "📊 Evaluation Results",
        "ℹ️ About"
    ])

# ════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("⚡ Federated Learning IDS")
    st.subheader("Privacy-preserving Intrusion Detection for Smart Energy IoT")
    st.divider()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Test Accuracy",  "94.65%", "↑ strong")
    col2.metric("Macro F1",       "68.40%", "DoS limited data")
    col3.metric("Normal F1",      "100.0%", "↑ perfect")
    col4.metric("Other F1",       "95.8%",  "↑ excellent")

    st.divider()
    st.markdown("### 🏗️ System Architecture")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
**Federated Clients (Smart Energy IoT):**
- 🏠 Smart Home
- 🚗 EV Charging Station
- ⚡ Grid Sensor
- ☀️ Solar / Wind Controller
- 🏭 Industrial Energy System
        """)
    with col2:
        st.markdown("""
**Deep Learning Model:**
- Conv1D CNN Branch (local patterns)
- Bidirectional LSTM Branch (temporal)
- Attention Layer (key traffic focus)
- Softmax output (multi-class)
        """)

    st.divider()
    st.markdown("### 🎯 Detected Attack Classes")
    cols = st.columns(len(class_names))
    colors = ["🔴", "🟠", "🟢", "🔵"]
    for i, cls in enumerate(class_names):
        cols[i].info(f"{colors[i]} **{cls}**")

# ════════════════════════════════════════════════════════════════════
# PAGE 2 — LIVE DETECTION
# ════════════════════════════════════════════════════════════════════
elif page == "🔍 Live Detection":
    st.title("🔍 Live Traffic Detection")
    st.caption("Upload a CSV file of network traffic to detect attack types")

    uploaded = st.file_uploader(
        "Upload network traffic CSV",
        type=["csv"],
        help="CSV should contain numeric network traffic features"
    )

    if uploaded:
        df = pd.read_csv(uploaded)
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Total rows: {len(df)} | Columns: {len(df.columns)}")

        numeric = df.select_dtypes(include=np.number)

        if len(numeric.columns) == 0:
            st.error("No numeric columns found in the file!")
        elif len(numeric) < WINDOW:
            st.warning(f"Need at least {WINDOW} rows. Got {len(numeric)}.")
        else:
            with st.spinner("Running detection..."):
                # Pad or trim features to match model input
                X = numeric.values.astype(np.float32)
                n_model_features = 46
                if X.shape[1] < n_model_features:
                    pad = np.zeros((X.shape[0],
                                   n_model_features - X.shape[1]),
                                   dtype=np.float32)
                    X = np.concatenate([X, pad], axis=1)
                elif X.shape[1] > n_model_features:
                    X = X[:, :n_model_features]

                # Sliding window predictions
                preds, confs, windows = [], [], []
                for i in range(0, len(X) - WINDOW, WINDOW):
                    seq = X[i : i + WINDOW][np.newaxis]
                    proba = model.predict(seq, verbose=0)[0]
                    preds.append(class_names[np.argmax(proba)])
                    confs.append(float(np.max(proba)))
                    windows.append(i)

            result_df = pd.DataFrame({
                "Window Start Row": windows,
                "Predicted Attack": preds,
                "Confidence": [f"{c:.2%}" for c in confs],
            })

            # ── Summary metrics ───────────────────────────────────
            st.divider()
            st.subheader("📊 Detection Summary")
            attacks = [p for p in preds if p != "Normal"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Windows", len(preds))
            c2.metric("Attacks Detected", len(attacks))
            c3.metric("Normal Traffic",
                      len(preds) - len(attacks))
            c4.metric("Avg Confidence",
                      f"{np.mean(confs):.2%}")

            # ── Alert ─────────────────────────────────────────────
            if len(attacks) > 0:
                attack_types = list(set(attacks))
                st.error(
                    f"🚨 **ALERT: {len(attacks)} attack windows detected!** "
                    f"Types: {', '.join(attack_types)}"
                )
            else:
                st.success("✅ No attacks detected — traffic appears normal")

            # ── Results table ─────────────────────────────────────
            st.divider()
            st.subheader("📋 Window-by-window Results")
            st.dataframe(result_df, use_container_width=True)

            # ── Attack distribution chart ─────────────────────────
            st.divider()
            st.subheader("📈 Attack Distribution")
            dist = pd.Series(preds).value_counts()
            st.bar_chart(dist)

            # ── Download results ──────────────────────────────────
            csv = result_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Results CSV",
                csv,
                "detection_results.csv",
                "text/csv"
            )
    else:
        st.info("👆 Upload a CSV file to start detecting attacks")

        st.markdown("### 📌 How to test")
        st.markdown("""
1. Use any CSV with numeric network traffic columns
2. The model will slide a window of 20 rows at a time
3. Each window gets a predicted attack class + confidence score
4. You can use your test data from `saved_models/X_test.npy`
        """)

        # Quick test with saved test data
        if st.button("🧪 Run Quick Test with Saved Test Data"):
            with st.spinner("Loading test data..."):
                X_test = np.load("saved_models/X_test.npy")
                y_test = np.load("saved_models/y_test.npy")

            sample = X_test[:100]
            preds, confs = [], []
            for i in range(0, len(sample) - WINDOW, WINDOW):
                seq = sample[i:i+WINDOW][np.newaxis]
                proba = model.predict(seq, verbose=0)[0]
                preds.append(class_names[np.argmax(proba)])
                confs.append(float(np.max(proba)))

            st.success(f"✅ Ran {len(preds)} predictions on test data")
            result_df = pd.DataFrame({
                "Window": range(len(preds)),
                "Predicted": preds,
                "Confidence": [f"{c:.2%}" for c in confs]
            })
            st.dataframe(result_df, use_container_width=True)
            st.bar_chart(pd.Series(preds).value_counts())

# ════════════════════════════════════════════════════════════════════
# PAGE 3 — EVALUATION RESULTS
# ════════════════════════════════════════════════════════════════════
elif page == "📊 Evaluation Results":
    st.title("📊 Model Evaluation Results")
    results = load_eval_results()

    if results:
        st.subheader("Overall Metrics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy",  f"{results['accuracy']*100:.2f}%")
        c2.metric("Macro F1",  f"{results['macro_f1']*100:.2f}%")
        c3.metric("Precision", f"{results['precision']*100:.2f}%")
        c4.metric("Recall",    f"{results['recall']*100:.2f}%")

        st.divider()
        st.subheader("Per-class Metrics")
        per_df = pd.DataFrame({
            "Class":     results["classes"],
            "Precision": [f"{v*100:.1f}%" for v in results["per_p"]],
            "Recall":    [f"{v*100:.1f}%" for v in results["per_r"]],
            "F1 Score":  [f"{v*100:.1f}%" for v in results["per_f1"]],
        })
        st.dataframe(per_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Per-class F1 Score Chart")
        f1_df = pd.DataFrame({
            "F1 Score": results["per_f1"]
        }, index=results["classes"])
        st.bar_chart(f1_df)

        st.divider()
        st.subheader("Confusion Matrix")
        try:
            st.image("evaluation/confusion_matrix.png",
                     caption="Confusion Matrix — CNN+BiLSTM+Attention IDS",
                     use_container_width=True)
        except:
            st.warning("Run evaluate.py first to generate the confusion matrix image")

    else:
        st.warning("No results found. Run `python -u evaluation/evaluate.py` first.")

# ════════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT
# ════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
### Federated Learning IDS for Smart Energy IoT

This system uses **Federated Learning** combined with a hybrid
**CNN + BiLSTM + Attention** deep learning model to detect
cyberattacks in Smart Energy IoT environments — without sharing
raw data between clients.

---

### Datasets
| Dataset | Source | Rows |
|---|---|---|
| CICIoT2023 | University of New Brunswick | 1,352,000 |
| EdgeIIoTset | IEEE DataPort | 69,387 |

---

### Model Architecture
| Component | Details |
|---|---|
| CNN Branch | Conv1D × 2, MaxPool, BatchNorm, Dropout |
| BiLSTM Branch | Bidirectional LSTM × 2, return_sequences=True |
| Attention | Scaled dot-product attention |
| Output | Dense → Softmax (4 classes) |

---

### Training Setup
| Parameter | Value |
|---|---|
| Window size | 20 |
| Stride | 5 |
| Optimizer | Adam (lr=0.001) |
| Batch size | 64 |
| Early stopping | patience=3 |
| Best epoch | 5 |

---

### Results
| Metric | Value |
|---|---|
| Test Accuracy | 94.65% |
| Macro F1 | 68.40% |
| Normal F1 | 100.0% |
| DDoS F1 | 77.8% |
| Other F1 | 95.8% |
    """)