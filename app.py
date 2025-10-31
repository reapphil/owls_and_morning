import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from filelock import FileLock, Timeout
import matplotlib.pyplot as plt
import io
import base64
import time

# -----------------------------
# Setup
# -----------------------------
DATA_FILE = "responses.csv"
LOCK_FILE = DATA_FILE + ".lock"
st.set_page_config(page_title="Morning vs. Night AI Demo", page_icon="🧠", layout="wide")

# -----------------------------
# Helpers
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

def load_data():
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=[
            "timestamp","wake_time","bed_time","coffee","energy","label"
        ]).to_csv(DATA_FILE, index=False)
    return ensure_columns(pd.read_csv(DATA_FILE))

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

    wake = st.slider("Wake-up time (hour, 3 AM – 12 PM)", 3, 12, 7)
    bed = st.slider("Bedtime (hour, 6 PM – 12 AM)", 18, 24, 23)
    coffee = st.slider("Cups of coffee/tea per day", 0, 10, 1)
    energy = st.slider("Morning energy (1–10)", 1, 10, 6)
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
        time.sleep(1)
        st.experimental_rerun()

# -----------------------------
# RESULTS PAGE
# -----------------------------
elif mode == "results":
    st.title("📊 Morning vs. Night — Results")
    st.caption("Auto-refreshes every 20 seconds as new entries arrive.")
    st.session_state.setdefault("last_refresh", time.time())

    # Refresh only every 20s
    if time.time() - st.session_state["last_refresh"] > 20:
        st.session_state["last_refresh"] = time.time()
        st.experimental_rerun()

    # Clear button
    if st.button("🗑️ Clear all responses"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        st.success("All responses deleted. Waiting for new submissions...")
        st.stop()

    df = load_data()
    if df.empty or df["label"].isna().all():
        st.info("No data yet — collect some responses first.")
        st.stop()

    df = df.dropna(subset=["wake_time", "bed_time", "label"])
    X = df[["wake_time", "bed_time"]].astype(float).values
    y = df["label"].astype(int).values

    if len(np.unique(y)) < 2:
        st.warning("Need at least one Morning Person and one Night Owl to draw a boundary.")
        st.stop()

    # Sidebar model selector
    st.sidebar.header("⚙️ Model Settings")
    model_name = st.sidebar.selectbox(
        "Select model type:",
        ["Logistic Regression", "k-Nearest Neighbors", "Decision Tree"],
        index=0
    )

    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "k-Nearest Neighbors":
        k = st.sidebar.slider("Number of neighbors (k)", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k)
    else:
        depth = st.sidebar.slider("Max depth", 1, 10, 3)
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model.fit(Xs, y)

    # Decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    Z = model.predict_proba(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))[:, 1].reshape(xx.shape)

    # Plot at 50% width
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.contourf(xx, yy, Z, levels=30, cmap="coolwarm", alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k", s=70)
    ax.set_xlabel("Wake-up time (3–12)")
    ax.set_ylabel("Bedtime (18–24)")
    ax.set_title(f"Decision Boundary — {model_name}")
    render_matplotlib(fig, width_pct=50)

    # Stats
    st.markdown("---")
    st.subheader("📈 Model Stats")
    st.write(f"👥 {len(y)} responses collected.")
    if len(y) >= 6:
        split = int(0.8 * len(y))
        Xtr, Xte = Xs[:split], Xs[split:]
        ytr, yte = y[:split], y[split:]
        acc = accuracy_score(yte, model.fit(Xtr, ytr).predict(Xte))
        st.metric("Holdout accuracy", f"{acc*100:.1f}%")
    else:
        st.write("Need more responses to estimate accuracy.")

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
