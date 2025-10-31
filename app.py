import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from filelock import FileLock, Timeout
import matplotlib.pyplot as plt
import io
import base64

# -----------------------------
# Setup
# -----------------------------
DATA_FILE = "responses.csv"
LOCK_FILE = DATA_FILE + ".lock"
st.set_page_config(page_title="Morning vs. Night AI Demo", page_icon="🧠", layout="wide")

# -----------------------------
# Utility functions
# -----------------------------
def ensure_columns(df):
    expected = ["timestamp", "wake_time", "bed_time", "coffee", "energy", "label"]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    return df[expected]

def append_row(row):
    lock = FileLock(LOCK_FILE)
    try:
        with lock.acquire(timeout=3):
            if not os.path.exists(DATA_FILE):
                pd.DataFrame(columns=row.keys()).to_csv(DATA_FILE, index=False)
            df = pd.read_csv(DATA_FILE)
            df = ensure_columns(df)
            df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            df.to_csv(DATA_FILE, index=False)
    except Timeout:
        st.error("File is busy — please try again in a few seconds.")

def generate_sample_data(n=100):
    """Generate realistic overlapping sample data — moderately fuzzy, not linearly separable."""
    np.random.seed(42)
    n_half = n // 2

    # Morning people: earlier wake/bed, moderate coffee, higher energy
    morning = pd.DataFrame({
        "timestamp": [datetime.now(timezone.utc).isoformat()] * n_half,
        "wake_time": np.clip(np.random.normal(0.35, 0.15, n_half), 0, 1),
        "bed_time": np.clip(np.random.normal(0.4, 0.15, n_half), 0, 1),
        "coffee": np.clip(np.random.normal(0.45, 0.2, n_half), 0, 1),
        "energy": np.clip(np.random.normal(0.7, 0.2, n_half), 0, 1),
        "label": [1] * n_half,  # 1 = morning
    })

    # Night owls: later wake/bed, more coffee, lower morning energy
    night = pd.DataFrame({
        "timestamp": [datetime.now(timezone.utc).isoformat()] * n_half,
        "wake_time": np.clip(np.random.normal(0.65, 0.15, n_half), 0, 1),
        "bed_time": np.clip(np.random.normal(0.65, 0.15, n_half), 0, 1),
        "coffee": np.clip(np.random.normal(0.6, 0.2, n_half), 0, 1),
        "energy": np.clip(np.random.normal(0.45, 0.2, n_half), 0, 1),
        "label": [0] * n_half,  # 0 = night
    })

    df = pd.concat([morning, night], ignore_index=True)

    # Add mild correlation and noise to make clusters natural
    df["wake_time"] = np.clip(
        df["wake_time"] * 0.7 + df["bed_time"] * 0.3 + np.random.normal(0, 0.05, n),
        0, 1
    )

    # Add small random label noise (~5%)
    flip_mask = np.random.rand(n) < 0.05
    df.loc[flip_mask, "label"] = 1 - df.loc[flip_mask, "label"]

    # Shuffle data for randomness
    df = df.sample(frac=1, random_state=99).reset_index(drop=True)
    return df



def load_data():
    if not os.path.exists(DATA_FILE):
        df = generate_sample_data()
        df.to_csv(DATA_FILE, index=False)
        return df
    df = pd.read_csv(DATA_FILE)
    if df.empty:
        df = generate_sample_data()
        df.to_csv(DATA_FILE, index=False)
    return ensure_columns(df)

def get_mode():
    try:
        qp = st.query_params
    except Exception:
        qp = st.experimental_get_query_params()
    val = qp.get("mode", "input")
    if isinstance(val, list):
        val = val[0] if val else "input"
    return str(val).lower().strip()

def render_matplotlib(fig, width_pct=50):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    html = f"""
    <div style='display:flex;justify-content:center;'>
        <img src='data:image/png;base64,{encoded}' style='width:{width_pct}%;height:auto;border-radius:10px;box-shadow:0 0 10px rgba(0,0,0,0.1);'/>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    plt.close(fig)

# -----------------------------
# Determine mode
# -----------------------------
mode = get_mode()

# -----------------------------
# INPUT PAGE
# -----------------------------
if mode == "input":
    st.title("🌅 Morning vs. Night — Audience Input")

    wake = st.slider("Wake-up time (very early ⟶ very late)", 0.0, 1.0, 0.5)
    bed = st.slider("Bedtime (very early ⟶ very late)", 0.0, 1.0, 0.5)
    coffee = st.slider("Coffee/tea consumption (no coffee ⟶ a lot of coffee)", 0.0, 1.0, 0.5)
    energy = st.slider("Morning energy (very low ⟶ very high)", 0.0, 1.0, 0.5)
    label = st.radio("Are you a morning person?", ["No", "Yes"], horizontal=True)

    if st.button("Submit ✅"):
        append_row({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "wake_time": wake,
            "bed_time": bed,
            "coffee": coffee,
            "energy": energy,
            "label": 1 if label == "Yes" else 0,
        })
        st.success("✅ Submitted! Thank you for participating!")

# -----------------------------
# RESULTS PAGE
# -----------------------------
elif mode == "results":
    st.title("📊 Morning vs. Night — Results")

    if st.button("🗑️ Clear all responses"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        st.success("All responses deleted. Synthetic data will repopulate on next load.")
        st.stop()

    df = load_data()
    df = df.dropna(subset=["wake_time", "bed_time", "label"])
    X = df[["wake_time", "bed_time"]].astype(float).values
    y = df["label"].astype(int).values

    # Sidebar model selector
    st.sidebar.header("⚙️ Model Settings")
    model_name = st.sidebar.selectbox(
        "Select model type:",
        ["Logistic Regression", "k-Nearest Neighbors", "Decision Tree", "Neural Network"],
        index=0
    )

    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "k-Nearest Neighbors":
        k = st.sidebar.slider("Number of neighbors (k)", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k)
    elif model_name == "Decision Tree":
        depth = st.sidebar.slider("Max depth", 1, 10, 3)
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    else:
        hidden = st.sidebar.slider("Hidden layer size", 2, 50, 10)
        model = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=2000, random_state=42)

    # Train and plot
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model.fit(Xs, y)

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = model.predict_proba(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.contourf(xx, yy, Z, levels=30, cmap="coolwarm", alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k", s=70)
    ax.set_xlabel("Wake-up time (early ⟶ late)")
    ax.set_ylabel("Bedtime (early ⟶ late)")
    ax.set_title(f"Decision Boundary — {model_name}")
    render_matplotlib(fig, width_pct=50)

    st.markdown("---")
    st.subheader("📈 Model Stats")
    st.write(f"👥 {len(y)} responses collected.")
    if len(y) >= 6:
        split = int(0.8 * len(y))
        Xtr, Xte = Xs[:split], Xs[split:]
        ytr, yte = y[:split], y[split:]
        acc = accuracy_score(yte, model.fit(Xtr, ytr).predict(Xte))
        st.metric("Holdout accuracy", f"{acc*100:.1f}%")

    st.download_button(
        "⬇️ Download responses (CSV)",
        df.to_csv(index=False).encode(),
        "responses.csv",
        "text/csv",
    )

# -----------------------------
# FALLBACK
# -----------------------------
else:
    st.error("Unknown mode. Use `?mode=input` or `?mode=results` in the URL.")
