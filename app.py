import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from filelock import FileLock
import matplotlib.pyplot as plt
import time

# -----------------------------
# Basic setup
# -----------------------------
DATA_FILE = "responses.csv"
LOCK_FILE = DATA_FILE + ".lock"

st.set_page_config(page_title="Morning vs. Night AI Demo", page_icon="🧠", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def append_row(row: dict):
    """Append one new response row safely to the CSV."""
    lock = FileLock(LOCK_FILE)
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=row.keys()).to_csv(DATA_FILE, index=False)
    with lock:
        df = pd.read_csv(DATA_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)

def load_data():
    """Load CSV or create an empty one if it doesn’t exist."""
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=[
            "timestamp","wake_time","bed_time","coffee","energy","label"
        ]).to_csv(DATA_FILE, index=False)
    return pd.read_csv(DATA_FILE)

# -----------------------------
# URL parameter handling
# -----------------------------
params = st.query_params
mode = params.get("mode", ["input"])[0].lower()

# -----------------------------
# INPUT MODE
# -----------------------------
if mode == "input":
    st.title("🌅 Morning vs. Night — Audience Input")
    st.write("Submit your preferences! The results page (`?mode=results`) will visualize how the AI separates morning people from night owls.")
    st.caption("👉 After submitting, open the same link with `?mode=results` to see your point appear.")

    wake = st.number_input("Wake-up time (0–23)", 0, 23, 7)
    bed = st.number_input("Bedtime (0–23)", 0, 23, 23)
    coffee = st.slider("Cups of coffee/tea per day", 0, 10, 1)
    energy = st.slider("Morning energy (1–10)", 1, 10, 6)
    label = st.radio("Do you consider yourself a morning person?", ["No", "Yes"], horizontal=True)

    if st.button("Submit ✅"):
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "wake_time": wake,
            "bed_time": bed,
            "coffee": coffee,
            "energy": energy,
            "label": 1 if label == "Yes" else 0,
        }
        append_row(row)
        st.success("✅ Submitted! Your data is saved. Switch to `?mode=results` to see it plotted.")

# -----------------------------
# RESULTS MODE
# -----------------------------
elif mode == "results":
    st.title("📊 Morning vs. Night — Results")
    st.caption("This page auto-refreshes every 5 seconds while new data comes in.")

    # Auto-refresh every 5 seconds
    st_autorefresh = st.empty()
    st_autorefresh.info("Auto-refresh enabled (5 s)…")
    st.experimental_rerun  # ensure compatibility on older versions
    st_autorefresh = st.experimental_rerun if hasattr(st, "experimental_rerun") else None

    st_autorefresh_container = st.empty()
    refresh_counter = st_autorefresh_container.experimental_rerun if hasattr(st_autorefresh_container, "experimental_rerun") else None
    st_autorefresh_counter = st.experimental_rerun if hasattr(st, "experimental_rerun") else None
    st_autorefresh_counter = None  # just a placeholder

    st_autorefresh = st.experimental_rerun if hasattr(st, "experimental_rerun") else None
    # Streamlit built-in timer
    st_autorefresh = st.experimental_rerun if hasattr(st, "experimental_rerun") else None
    st_autorefresh = st_autorefresh  # dummy to silence lint

    # Real refresh every 5 seconds
    st_autorefresh = st.empty()
    st_autorefresh.markdown(
        """
        <meta http-equiv="refresh" content="5">
        """,
        unsafe_allow_html=True,
    )

    df = load_data()
    if df.empty:
        st.info("No data yet — switch to `?mode=input` and submit a few entries.")
        st.stop()

    X = df[["wake_time", "bed_time"]].astype(float).values
    y = df["label"].astype(int).values

    if len(np.unique(y)) < 2:
        st.warning("Need at least one Morning Person and one Night Owl to draw a boundary.")
        st.stop()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression()
    clf.fit(Xs, y)

    # Create decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict_proba(scaler.transform(grid))[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, Z, levels=30, cmap="coolwarm", alpha=0.3)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k", s=80)
    ax.set_xlabel("Wake-up time (0–23)")
    ax.set_ylabel("Bedtime (0–23)")
    ax.set_title("Decision Boundary — Morning (blue) vs. Night (red)")
    st.pyplot(fig)

    # Quick stats
    st.markdown("---")
    st.subheader("📈 Model Stats")
    if len(y) >= 6:
        split = int(0.8 * len(y))
        Xtr, Xte = Xs[:split], Xs[split:]
        ytr, yte = y[:split], y[split:]
        clf2 = LogisticRegression().fit(Xtr, ytr)
        acc = accuracy_score(yte, clf2.predict(Xte))
        st.metric("Holdout accuracy", f"{acc*100:.1f}%")
    else:
        st.write("Need a few more responses to estimate accuracy.")

    st.download_button(
        "⬇️ Download responses (CSV)",
        df.to_csv(index=False).encode("utf-8"),
        "responses.csv",
        "text/csv",
    )

# -----------------------------
# ERROR MODE
# -----------------------------
else:
    st.error("Unknown mode. Use `?mode=input` or `?mode=results` in the URL.")
