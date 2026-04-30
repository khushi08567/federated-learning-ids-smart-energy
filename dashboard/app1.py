import sys
sys.stdout.reconfigure(line_buffering=True)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── Page config ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="FL IDS — Smart Energy IoT",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS — Pastel Theme ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

/* ── Root palette ── */
:root {
  --mint:       #C8E6C9;
  --mint-dark:  #81C784;
  --lavender:   #E1D5F0;
  --lav-dark:   #B39DDB;
  --peach:      #FFE0B2;
  --peach-dark: #FFAB76;
  --sky:        #BBDEFB;
  --sky-dark:   #64B5F6;
  --rose:       #FADADD;
  --rose-dark:  #F48FB1;
  --cream:      #FAFAF7;
  --charcoal:   #3D3D3D;
  --muted:      #888;
}

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background-color: var(--cream) !important;
  color: var(--charcoal) !important;
}

/* Main background */
.stApp { background: #F7F4EF !important; }
.main .block-container { padding: 2rem 2.5rem !important; max-width: 1200px; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(160deg, #EDE7F6 0%, #E8F5E9 100%) !important;
  border-right: 1px solid #DDD !important;
}
[data-testid="stSidebar"] * { color: var(--charcoal) !important; }
[data-testid="stSidebar"] hr { border-color: #CCC !important; }

/* Radio buttons in sidebar */
[data-testid="stSidebar"] .stRadio label {
  background: white !important;
  border-radius: 10px !important;
  padding: 8px 14px !important;
  margin: 3px 0 !important;
  border: 1px solid #E0E0E0 !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: #EDE7F6 !important;
  border-color: var(--lav-dark) !important;
}

/* Metrics */
[data-testid="metric-container"] {
  background: white !important;
  border-radius: 16px !important;
  padding: 1.2rem !important;
  border: 1px solid #EEE !important;
  box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important;
}
[data-testid="stMetricLabel"]  { color: var(--muted) !important; font-size: 13px !important; }
[data-testid="stMetricValue"]  { color: var(--charcoal) !important; font-size: 28px !important; font-weight: 600 !important; }
[data-testid="stMetricDelta"]  { font-size: 12px !important; }

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, #B39DDB, #81C784) !important;
  color: white !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 10px 24px !important;
  font-weight: 500 !important;
  transition: all 0.3s !important;
  box-shadow: 0 4px 15px rgba(179,157,219,0.4) !important;
}
.stButton > button:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 6px 20px rgba(179,157,219,0.5) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
  background: white !important;
  border-radius: 16px !important;
  border: 2px dashed #B39DDB !important;
  padding: 1rem !important;
}

/* Dataframe */
[data-testid="stDataFrame"] {
  border-radius: 12px !important;
  overflow: hidden !important;
  border: 1px solid #EEE !important;
}

/* Alerts */
.stSuccess { background: #E8F5E9 !important; border-radius: 12px !important; border-left: 4px solid #81C784 !important; }
.stError   { background: #FADADD !important; border-radius: 12px !important; border-left: 4px solid #F48FB1 !important; }
.stInfo    { background: #E3F2FD !important; border-radius: 12px !important; border-left: 4px solid #64B5F6 !important; }
.stWarning { background: #FFF8E1 !important; border-radius: 12px !important; border-left: 4px solid #FFD54F !important; }

/* Divider */
hr { border-color: #EEE !important; }

/* Bar chart */
[data-testid="stVegaLiteChart"] { border-radius: 12px !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: white !important;
  border-radius: 12px !important;
  padding: 4px !important;
  border: 1px solid #EEE !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: 10px !important;
  font-weight: 500 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Helpers ───────────────────────────────────────────────────────────
def pastel_card(title, value, subtitle, color):
    st.markdown(f"""
    <div style="background:{color};border-radius:16px;padding:20px 22px;
                border:1px solid rgba(0,0,0,0.06);
                box-shadow:0 2px 12px rgba(0,0,0,0.05)">
      <p style="margin:0;font-size:12px;color:#666;text-transform:uppercase;
                letter-spacing:.06em;font-weight:500">{title}</p>
      <p style="margin:6px 0 4px;font-size:30px;font-weight:600;color:#3D3D3D">{value}</p>
      <p style="margin:0;font-size:12px;color:#888">{subtitle}</p>
    </div>""", unsafe_allow_html=True)

def section_header(icon, title, subtitle=""):
    st.markdown(f"""
    <div style="margin:1.5rem 0 1rem">
      <h3 style="margin:0;font-family:'DM Serif Display',serif;
                 font-size:22px;color:#3D3D3D">{icon} {title}</h3>
      {"<p style='margin:4px 0 0;font-size:13px;color:#888'>"+subtitle+"</p>" if subtitle else ""}
    </div>""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model       = tf.keras.models.load_model("saved_models/best_model.h5")
    class_names = np.load("saved_models/class_names.npy", allow_pickle=True)
    return model, [str(c) for c in class_names]

@st.cache_data
def load_results():
    try:
        with open("evaluation/results.json") as f:
            return json.load(f)
    except:
        return None

model, class_names = load_model()
WINDOW = 20
CLASS_COLORS = {
    "DDoS":   "#BBDEFB",
    "DoS":    "#FADADD",
    "Normal": "#C8E6C9",
    "Other":  "#FFE0B2",
}

# ── Sidebar ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
      <div style="font-size:48px">⚡</div>
      <h2 style="margin:8px 0 4px;font-family:'DM Serif Display',serif;
                 font-size:20px;color:#3D3D3D">FL IDS System</h2>
      <p style="margin:0;font-size:12px;color:#888">Smart Energy IoT</p>
    </div>""", unsafe_allow_html=True)

    st.divider()

    page = st.radio("", [
        "🏠  Home",
        "🔍  Live Detection",
        "📊  Evaluation",
        "🏗️  Architecture",
        "ℹ️  About",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown("""
    <div style="font-size:12px;color:#999;line-height:1.7">
      <b style="color:#666">Model</b><br>CNN + BiLSTM + Attention<br><br>
      <b style="color:#666">Datasets</b><br>CICIoT2023 · EdgeIIoTset<br><br>
      <b style="color:#666">Accuracy</b><br>94.65% on test set
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# PAGE — HOME
# ════════════════════════════════════════════════════════════════════
if "Home" in page:
    st.markdown("""
    <div style="padding:2rem 0 1rem">
      <p style="margin:0;font-size:13px;color:#B39DDB;font-weight:600;
                letter-spacing:.1em;text-transform:uppercase">
        Final Year Major Project</p>
      <h1 style="margin:6px 0 8px;font-family:'DM Serif Display',serif;
                 font-size:40px;color:#3D3D3D;line-height:1.15">
        Federated Learning<br>Intrusion Detection System</h1>
      <p style="margin:0;font-size:15px;color:#888;max-width:600px">
        Privacy-preserving cyberattack detection for Smart Energy IoT
        environments using a hybrid CNN + BiLSTM + Attention architecture.
      </p>
    </div>""", unsafe_allow_html=True)

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    with c1: pastel_card("Test Accuracy",  "94.65%", "on held-out test set",   "#E8F5E9")
    with c2: pastel_card("Macro F1",       "68.40%", "avg across all classes", "#EDE7F6")
    with c3: pastel_card("Normal F1",      "100.0%", "perfect detection",      "#E3F2FD")
    with c4: pastel_card("DDoS F1",        "77.8%",  "good detection",         "#FFF8E1")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        section_header("🌐", "Federated IoT Clients",
                       "5 simulated smart energy environments")
        clients = [
            ("🏠", "Smart Home",            "#E8F5E9"),
            ("🚗", "EV Charging Station",   "#E3F2FD"),
            ("⚡", "Grid Sensor",           "#FFF8E1"),
            ("☀️", "Solar/Wind Controller", "#EDE7F6"),
            ("🏭", "Industrial Energy",     "#FADADD"),
        ]
        for icon, name, color in clients:
            st.markdown(f"""
            <div style="background:{color};border-radius:12px;padding:12px 16px;
                        margin:6px 0;border:1px solid rgba(0,0,0,0.05);
                        display:flex;align-items:center;gap:12px">
              <span style="font-size:20px">{icon}</span>
              <span style="font-weight:500;color:#3D3D3D">{name}</span>
            </div>""", unsafe_allow_html=True)

    with col2:
        section_header("🎯", "Detected Attack Classes",
                       "Multi-class classification output")
        classes_info = [
            ("DDoS",   "#BBDEFB", "Distributed Denial of Service"),
            ("DoS",    "#FADADD", "Denial of Service"),
            ("Normal", "#C8E6C9", "Benign traffic"),
            ("Other",  "#FFE0B2", "Other attack types"),
        ]
        for cls, color, desc in classes_info:
            st.markdown(f"""
            <div style="background:{color};border-radius:12px;padding:14px 18px;
                        margin:6px 0;border:1px solid rgba(0,0,0,0.06)">
              <p style="margin:0;font-weight:600;color:#3D3D3D;font-size:15px">{cls}</p>
              <p style="margin:3px 0 0;font-size:12px;color:#666">{desc}</p>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# PAGE — LIVE DETECTION
# ════════════════════════════════════════════════════════════════════
elif "Detection" in page:
    st.markdown("""
    <h1 style="font-family:'DM Serif Display',serif;font-size:36px;
               color:#3D3D3D;margin-bottom:6px">🔍 Live Traffic Detection</h1>
    <p style="color:#888;font-size:14px;margin-bottom:2rem">
      Upload a CSV file of network traffic to detect attack types in real time</p>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"],
                                label_visibility="collapsed")

    if not uploaded:
        st.markdown("""
        <div style="background:#EDE7F6;border-radius:16px;padding:2rem;
                    text-align:center;border:2px dashed #B39DDB;margin:1rem 0">
          <div style="font-size:40px;margin-bottom:12px">📂</div>
          <p style="font-weight:600;color:#3D3D3D;margin:0">
            Drop your network traffic CSV here</p>
          <p style="color:#888;font-size:13px;margin:6px 0 0">
            Supports any CSV with numeric network traffic features</p>
        </div>""", unsafe_allow_html=True)

        if st.button("🧪 Quick Test with Saved Test Data"):
            with st.spinner("Running predictions..."):
                X_test = np.load("saved_models/X_test.npy")
                sample = X_test[:200]
                preds, confs = [], []
                for i in range(0, len(sample) - WINDOW, WINDOW):
                    seq   = sample[i:i+WINDOW][np.newaxis]
                    proba = model.predict(seq, verbose=0)[0]
                    preds.append(class_names[np.argmax(proba)])
                    confs.append(float(np.max(proba)))

            c1, c2, c3, c4 = st.columns(4)
            with c1: pastel_card("Windows", str(len(preds)), "analyzed", "#E8F5E9")
            with c2: pastel_card("Attacks",
                                 str(sum(p!="Normal" for p in preds)),
                                 "detected", "#FADADD")
            with c3: pastel_card("Normal",
                                 str(sum(p=="Normal" for p in preds)),
                                 "traffic", "#E3F2FD")
            with c4: pastel_card("Avg Conf",
                                 f"{np.mean(confs):.1%}",
                                 "confidence", "#EDE7F6")

            st.markdown("<br>", unsafe_allow_html=True)
            result_df = pd.DataFrame({
                "Window": range(len(preds)),
                "Predicted Class": preds,
                "Confidence": [f"{c:.2%}" for c in confs]
            })
            st.dataframe(result_df, use_container_width=True, hide_index=True)

            dist = pd.Series(preds).value_counts().reset_index()
            dist.columns = ["Attack Class", "Count"]
            st.bar_chart(dist.set_index("Attack Class"))
    else:
        df = pd.read_csv(uploaded)
        st.markdown(f"""
        <div style="background:#E8F5E9;border-radius:12px;padding:12px 18px;
                    margin-bottom:1rem;border:1px solid #A5D6A7">
          ✅ &nbsp;<b>{uploaded.name}</b> loaded —
          {len(df):,} rows · {len(df.columns)} columns
        </div>""", unsafe_allow_html=True)

        with st.expander("👀 Preview data"):
            st.dataframe(df.head(10), use_container_width=True)

        numeric = df.select_dtypes(include=np.number)
        if len(numeric) < WINDOW:
            st.error(f"Need at least {WINDOW} rows. Got {len(numeric)}.")
        else:
            with st.spinner("🔍 Analyzing traffic..."):
                X = numeric.values.astype("float32")
                n = 46
                if X.shape[1] < n:
                    X = np.concatenate(
                        [X, np.zeros((X.shape[0], n-X.shape[1]), "float32")], 1)
                elif X.shape[1] > n:
                    X = X[:, :n]

                preds, confs, rows = [], [], []
                for i in range(0, len(X)-WINDOW, WINDOW):
                    proba = model.predict(X[i:i+WINDOW][np.newaxis], verbose=0)[0]
                    preds.append(class_names[np.argmax(proba)])
                    confs.append(float(np.max(proba)))
                    rows.append(i)

            attacks = [p for p in preds if p != "Normal"]
            st.markdown("<br>", unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1: pastel_card("Windows",   str(len(preds)),   "analyzed",  "#E8F5E9")
            with c2: pastel_card("Attacks",   str(len(attacks)), "detected",  "#FADADD")
            with c3: pastel_card("Normal",
                                 str(len(preds)-len(attacks)), "safe",     "#E3F2FD")
            with c4: pastel_card("Avg Conf",
                                 f"{np.mean(confs):.1%}", "confidence", "#EDE7F6")

            st.markdown("<br>", unsafe_allow_html=True)
            if attacks:
                st.error(f"🚨 **{len(attacks)} attack windows detected** — "
                         f"Types: {', '.join(set(attacks))}")
            else:
                st.success("✅ All traffic appears normal — no attacks detected")

            result_df = pd.DataFrame({
                "Window Start": rows,
                "Predicted Class": preds,
                "Confidence": [f"{c:.2%}" for c in confs]
            })
            st.dataframe(result_df, use_container_width=True, hide_index=True)
            st.bar_chart(pd.Series(preds).value_counts())

            csv = result_df.to_csv(index=False)
            st.download_button("⬇️ Download Results",
                               csv, "detection_results.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════
# PAGE — EVALUATION
# ════════════════════════════════════════════════════════════════════
elif "Evaluation" in page:
    st.markdown("""
    <h1 style="font-family:'DM Serif Display',serif;font-size:36px;
               color:#3D3D3D;margin-bottom:6px">📊 Model Evaluation</h1>
    <p style="color:#888;font-size:14px;margin-bottom:2rem">
      Complete performance metrics on held-out test data</p>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: pastel_card("Accuracy",  "94.65%", "overall correctness",        "#E8F5E9")
    with c2: pastel_card("Macro F1",  "68.40%", "avg across classes",         "#EDE7F6")
    with c3: pastel_card("Precision", "68.97%", "macro avg",                  "#E3F2FD")
    with c4: pastel_card("Recall",    "67.90%", "macro avg",                  "#FFF8E1")

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])

    with col1:
        section_header("📋", "Per-class Breakdown")
        per_data = {
            "Class":     ["DDoS",   "DoS",   "Normal", "Other"],
            "Precision": ["81.0%",  "0.0%",  "100.0%", "94.9%"],
            "Recall":    ["74.9%",  "0.0%",  "100.0%", "96.8%"],
            "F1 Score":  ["77.8%",  "0.0%",  "100.0%", "95.8%"],
            "Status":    ["✅ Good", "⚠️ Low data",
                          "🌟 Perfect", "✅ Excellent"],
        }
        st.dataframe(pd.DataFrame(per_data),
                     use_container_width=True, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        section_header("📈", "F1 Score by Class")
        f1_df = pd.DataFrame({
            "F1 Score": [0.778, 0.0, 1.0, 0.958]
        }, index=["DDoS","DoS","Normal","Other"])
        st.bar_chart(f1_df, color="#B39DDB")

    with col2:
        section_header("🗺️", "Confusion Matrix")
        try:
            st.image("evaluation/confusion_matrix.png",
                     use_container_width=True)
        except:
            fig, ax = plt.subplots(figsize=(6, 5))
            cm = np.array([
                [1233,  2,  0,  320],
                [   3,  0,  0,   20],
                [   0,  0,2775,   0],
                [  25,  0,  0, 6592]
            ])
            sns.heatmap(cm, annot=True, fmt="d",
                        cmap="PuBu",
                        xticklabels=class_names,
                        yticklabels=class_names, ax=ax)
            ax.set_title("Confusion Matrix", pad=12)
            ax.set_ylabel("True Label")
            ax.set_xlabel("Predicted Label")
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="background:#FFF8E1;border-radius:14px;padding:16px 20px;
                border-left:4px solid #FFD54F">
      <b>💡 Why is DoS F1 = 0%?</b><br>
      <span style="color:#666;font-size:13px">
        DoS class had only <b>147 sequences</b> in training vs 33,360 for Other.
        Increasing <code>ROWS_PER_CIC_FILE</code> in preprocess.py will fix this
        by capturing more DoS samples from the dataset.</span>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# PAGE — ARCHITECTURE
# ════════════════════════════════════════════════════════════════════
elif "Architecture" in page:
    st.markdown("""
    <h1 style="font-family:'DM Serif Display',serif;font-size:36px;
               color:#3D3D3D;margin-bottom:6px">🏗️ System Architecture</h1>
    <p style="color:#888;font-size:14px;margin-bottom:2rem">
      How the federated learning pipeline and deep learning model work</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:white;border-radius:20px;padding:2rem;
                border:1px solid #EEE;margin-bottom:1.5rem">
      <p style="font-weight:600;color:#3D3D3D;margin:0 0 1.5rem;font-size:16px">
        🔄 Federated Learning Pipeline</p>
      <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:12px">
        <div style="background:#EDE7F6;border-radius:12px;padding:14px;text-align:center">
          <div style="font-size:24px">📦</div>
          <p style="margin:6px 0 2px;font-weight:600;font-size:13px">Local Data</p>
          <p style="margin:0;font-size:11px;color:#888">stays on device</p>
        </div>
        <div style="background:#E3F2FD;border-radius:12px;padding:14px;text-align:center">
          <div style="font-size:24px">🧠</div>
          <p style="margin:6px 0 2px;font-weight:600;font-size:13px">Local Train</p>
          <p style="margin:0;font-size:11px;color:#888">each client trains</p>
        </div>
        <div style="background:#E8F5E9;border-radius:12px;padding:14px;text-align:center">
          <div style="font-size:24px">📤</div>
          <p style="margin:6px 0 2px;font-weight:600;font-size:13px">Send Weights</p>
          <p style="margin:0;font-size:11px;color:#888">not raw data</p>
        </div>
        <div style="background:#FFF8E1;border-radius:12px;padding:14px;text-align:center">
          <div style="font-size:24px">⚖️</div>
          <p style="margin:6px 0 2px;font-weight:600;font-size:13px">FedAvg</p>
          <p style="margin:0;font-size:11px;color:#888">server aggregates</p>
        </div>
        <div style="background:#FADADD;border-radius:12px;padding:14px;text-align:center">
          <div style="font-size:24px">🌐</div>
          <p style="margin:6px 0 2px;font-weight:600;font-size:13px">Global Model</p>
          <p style="margin:0;font-size:11px;color:#888">broadcast back</p>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        section_header("🧬", "Model Layers")
        layers = [
            ("Input",          "(20, 46)",  "#EDE7F6", "window × features"),
            ("Conv1D × 2",     "(None, 128)","#E3F2FD","CNN branch"),
            ("BiLSTM × 2",     "(None, 128)","#E8F5E9","temporal branch"),
            ("Attention",      "(None, 128)","#FFF8E1","focus mechanism"),
            ("Concatenate",    "(None, 256)","#FADADD","merge branches"),
            ("Dense → Softmax","(None, 4)",  "#EDE7F6","4-class output"),
        ]
        for name, shape, color, desc in layers:
            st.markdown(f"""
            <div style="background:{color};border-radius:10px;
                        padding:10px 16px;margin:5px 0;
                        display:flex;justify-content:space-between;
                        align-items:center;border:1px solid rgba(0,0,0,0.05)">
              <div>
                <span style="font-weight:600;font-size:13px">{name}</span>
                <span style="color:#888;font-size:12px;margin-left:8px">{desc}</span>
              </div>
              <code style="background:white;padding:2px 8px;border-radius:6px;
                           font-size:11px;color:#666">{shape}</code>
            </div>""", unsafe_allow_html=True)

    with col2:
        section_header("⚙️", "Hyperparameters")
        params = [
            ("Window Size",    "20 rows",     "#E8F5E9"),
            ("Stride",         "5 rows",      "#EDE7F6"),
            ("CNN Filters",    "64 → 128",    "#E3F2FD"),
            ("LSTM Units",     "64 × 2 BiDir","#FFF8E1"),
            ("Optimizer",      "Adam",        "#FADADD"),
            ("Learning Rate",  "0.001",       "#E8F5E9"),
            ("Batch Size",     "64",          "#EDE7F6"),
            ("Early Stopping", "patience = 3","#E3F2FD"),
            ("Best Epoch",     "5 / 20",      "#FFF8E1"),
            ("FL Rounds",      "10",          "#FADADD"),
        ]
        for name, val, color in params:
            st.markdown(f"""
            <div style="background:{color};border-radius:10px;
                        padding:10px 16px;margin:5px 0;
                        display:flex;justify-content:space-between;
                        border:1px solid rgba(0,0,0,0.05)">
              <span style="font-weight:500;font-size:13px">{name}</span>
              <span style="color:#555;font-size:13px">{val}</span>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════
# PAGE — ABOUT
# ════════════════════════════════════════════════════════════════════
elif "About" in page:
    st.markdown("""
    <h1 style="font-family:'DM Serif Display',serif;font-size:36px;
               color:#3D3D3D;margin-bottom:6px">ℹ️ About This Project</h1>
    <p style="color:#888;font-size:14px;margin-bottom:2rem">
      Final year major project details and dataset information</p>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style="background:white;border-radius:20px;padding:1.5rem;
                    border:1px solid #EEE;margin-bottom:1rem">
          <p style="font-weight:600;font-size:16px;margin:0 0 1rem">
            🎯 Project Goal</p>
          <p style="color:#666;font-size:14px;line-height:1.7;margin:0">
            Build a <b>privacy-preserving Intrusion Detection System (IDS)</b>
            for Smart Energy IoT environments using Federated Learning.
            Raw data never leaves the local device — only model weights
            are shared with the central server.
          </p>
        </div>

        <div style="background:#EDE7F6;border-radius:20px;padding:1.5rem;
                    border:1px solid #D1C4E9">
          <p style="font-weight:600;font-size:16px;margin:0 0 1rem">
            📚 Datasets Used</p>
          <div style="background:white;border-radius:10px;padding:12px 14px;
                      margin-bottom:8px">
            <p style="margin:0;font-weight:600;font-size:13px">CICIoT2023</p>
            <p style="margin:3px 0 0;font-size:12px;color:#888">
              University of New Brunswick · 1,352,000 rows · 46 features</p>
          </div>
          <div style="background:white;border-radius:10px;padding:12px 14px">
            <p style="margin:0;font-weight:600;font-size:13px">EdgeIIoTset</p>
            <p style="margin:3px 0 0;font-size:12px;color:#888">
              IEEE DataPort · 69,387 rows · 46 features</p>
          </div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background:#E8F5E9;border-radius:20px;padding:1.5rem;
                    border:1px solid #C8E6C9;margin-bottom:1rem">
          <p style="font-weight:600;font-size:16px;margin:0 0 1rem">
            🏆 Key Results</p>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
            <div style="background:white;border-radius:10px;padding:12px;
                        text-align:center">
              <p style="margin:0;font-size:24px;font-weight:700;
                         color:#3D3D3D">94.65%</p>
              <p style="margin:4px 0 0;font-size:11px;color:#888">Accuracy</p>
            </div>
            <div style="background:white;border-radius:10px;padding:12px;
                        text-align:center">
              <p style="margin:0;font-size:24px;font-weight:700;
                         color:#3D3D3D">100%</p>
              <p style="margin:4px 0 0;font-size:11px;color:#888">Normal F1</p>
            </div>
            <div style="background:white;border-radius:10px;padding:12px;
                        text-align:center">
              <p style="margin:0;font-size:24px;font-weight:700;
                         color:#3D3D3D">77.8%</p>
              <p style="margin:4px 0 0;font-size:11px;color:#888">DDoS F1</p>
            </div>
            <div style="background:white;border-radius:10px;padding:12px;
                        text-align:center">
              <p style="margin:0;font-size:24px;font-weight:700;
                         color:#3D3D3D">95.8%</p>
              <p style="margin:4px 0 0;font-size:11px;color:#888">Other F1</p>
            </div>
          </div>
        </div>

        <div style="background:#FFF8E1;border-radius:20px;padding:1.5rem;
                    border:1px solid #FFE082">
          <p style="font-weight:600;font-size:16px;margin:0 0 1rem">
            🔒 Privacy Guarantees</p>
          <div style="font-size:13px;color:#666;line-height:1.8">
            ✅ Raw data never leaves client devices<br>
            ✅ Only model weights shared<br>
            ✅ FedAvg aggregation on server<br>
            ✅ No central data collection<br>
            ✅ Supports non-IID data distributions
          </div>
        </div>""", unsafe_allow_html=True)