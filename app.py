import os
import io
import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import streamlit as st
from filelock import FileLock, Timeout
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# Setup
# -----------------------------
st.set_page_config(
    page_title="Morning vs. Night AI Demo",
    page_icon="🧠",
    layout="wide"
)

DATA_FILE = "responses.csv"
LOCK_FILE = DATA_FILE + ".lock"

# -----------------------------
# UTILITIES
# -----------------------------
def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    expected = ["timestamp", "wake_time", "bed_time", "coffee", "energy", "label"]
    for col in expected:
        if col not in df.columns:
            df[col] = np.nan
    return df[expected]


def append_row(row: dict) -> None:
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
        st.error("File is busy — please try again.")


def generate_sample_data(n=100):
    np.random.seed(62)
    n_half = n // 2

    morning = pd.DataFrame({
        "timestamp": [datetime.now(timezone.utc).isoformat()] * n_half,
        "wake_time": np.clip(np.random.normal(0.35, 0.15, n_half), 0, 1),
        "bed_time": np.clip(np.random.normal(0.40, 0.15, n_half), 0, 1),
        "coffee":   np.clip(np.random.normal(0.45, 0.20, n_half), 0, 1),
        "energy":   np.clip(np.random.normal(0.70, 0.20, n_half), 0, 1),
        "label":    [1] * n_half,
    })

    night = pd.DataFrame({
        "timestamp": [datetime.now(timezone.utc).isoformat()] * n_half,
        "wake_time": np.clip(np.random.normal(0.65, 0.15, n_half), 0, 1),
        "bed_time": np.clip(np.random.normal(0.65, 0.15, n_half), 0, 1),
        "coffee":   np.clip(np.random.normal(0.60, 0.20, n_half), 0, 1),
        "energy":   np.clip(np.random.normal(0.45, 0.20, n_half), 0, 1),
        "label":    [0] * n_half,
    })

    df = pd.concat([morning, night], ignore_index=True)

    n_total = len(df)
    df["wake_time"] = np.clip(
        df["wake_time"] * 0.7 + df["bed_time"] * 0.3 + np.random.normal(0, 0.05, n_total),
        0, 1,
    )

    # 5 % Label noise
    flip = np.random.rand(n_total) < 0.05
    df.loc[flip, "label"] = 1 - df.loc[flip, "label"]

    return df.sample(frac=1, random_state=99).reset_index(drop=True)


def load_data():
    df = generate_sample_data()
    df.to_csv(DATA_FILE, index=False)
    return ensure_columns(df)


def get_mode():
    try:
        qp = st.query_params
    except Exception:
        qp = st.experimental_get_query_params()
    m = qp.get("mode", "input")
    if isinstance(m, list):
        m = m[0]
    return str(m).lower().strip()


def render_matplotlib(fig, width_pct=70):
    buf = io.BytesIO()
    fig.patch.set_facecolor("none")
    for ax in fig.axes:
        ax.set_facecolor("none")

    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", transparent=True)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()

    st.markdown(
        f"""
        <div style='display:flex;justify-content:center;'>
            <img src='data:image/png;base64,{encoded}'
                 style='width:{width_pct}%;height:auto;'/>
        </div>
        """,
        unsafe_allow_html=True
    )
    plt.close(fig)


# -----------------------------
# MODE
# -----------------------------
mode = get_mode()

# -----------------------------
# INPUT PAGE
# -----------------------------
if mode == "input":
    st.title("🌅 Morning vs. Night — Audience Input")

    wake = st.slider("Wake-up time (very early → very late)", 0.0, 1.0, 0.5)
    bed = st.slider("Bedtime (very early → very late)", 0.0, 1.0, 0.5)
    coffee = st.slider("Coffee consumption (none → a lot)", 0.0, 1.0, 0.5)
    energy = st.slider("Morning energy (low → high)", 0.0, 1.0, 0.5)
    label = st.radio("Are you a morning person?", ["No", "Yes"], horizontal=True)

    if st.button("Submit"):
        append_row({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "wake_time": wake,
            "bed_time": bed,
            "coffee": coffee,
            "energy": energy,
            "label": 1 if label == "Yes" else 0,
        })
        st.success("Thanks!")

# -----------------------------
# RESULTS PAGE
# -----------------------------
elif mode == "results":
    st.markdown(
        "<h2 style='text-align:center'>📊 Morning (Blue) vs. Night (Red)</h2>",
        unsafe_allow_html=True,
    )

    df = load_data()
    df = df.dropna(subset=["wake_time", "bed_time", "label"])
    X = df[["wake_time", "bed_time"]].values
    y = df["label"].astype(int).values

    # ---- SIDEBAR ----
    st.sidebar.header("⚙️ Controls")

    if st.sidebar.button("🗑 Clear all responses"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        st.sidebar.success("Cleared. Reload page.")
        st.stop()

    random_trigger = st.sidebar.button("🎲 Generate Random Person")

    diagram_width = st.sidebar.slider("Diagram width (%)", 30, 100, 70)

    st.sidebar.subheader("Model")
    model_name = st.sidebar.selectbox(
        "Model",
        ["Log Reg", "kNN", "Tree", "Neural Net"]
    )

    # MODEL
    if model_name == "Log Reg":
        model = LogisticRegression()
    elif model_name == "kNN":
        k = st.sidebar.slider("k", 1, 15, 5)
        model = KNeighborsClassifier(k)
    elif model_name == "Tree":
        d = st.sidebar.slider("Depth", 1, 10, 3)
        model = DecisionTreeClassifier(max_depth=d)
    else:
        L = st.sidebar.slider("Layers", 1, 4, 2)
        N = st.sidebar.slider("Neurons", 2, 20, 8)
        model = MLPClassifier(
            hidden_layer_sizes=tuple([N]*L),
            max_iter=2000,
            random_state=42
        )

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model.fit(Xs, y)

    # Decision boundary
    x_min, x_max = X[:, 0].min()-0.1, X[:, 0].max()+0.1
    y_min, y_max = X[:, 1].min()-0.1, X[:, 1].max()+0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300),
    )
    Z = model.predict_proba(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))[:, 1]
    Z = Z.reshape(xx.shape)

    # -----------------------------
    # RANDOM PERSON — robust RNG
    # -----------------------------
    if "random_point" not in st.session_state:
        st.session_state.random_point = None
        st.session_state.prediction_text = None

    if random_trigger:
        rng2 = np.random.default_rng()          # independent RNG
        st.session_state.random_point = rng2.random((1, 2))
        pred = model.predict(
            scaler.transform(st.session_state.random_point)
        )[0]
        label_txt = "🌅 Morning" if pred == 1 else "🌙 Night"
        st.session_state.prediction_text = f"Model says: **{label_txt}**"

    random_point = st.session_state.random_point
    prediction_text = st.session_state.prediction_text

    # -----------------------------
    # PLOT
    # -----------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.contourf(xx, yy, Z, 30, cmap="coolwarm", alpha=0.25)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k", s=70)

    if random_point is not None:
        ax.scatter(
            random_point[:, 0], random_point[:, 1],
            color="limegreen", edgecolor="black", s=150,
            label="Random Person"
        )
        ax.legend()

    ax.set_xlabel("Wake-up time (early→late)")
    ax.set_ylabel("Bedtime (early→late)")
    ax.set_title(f"Decision Boundary — {model_name}")

    render_matplotlib(fig, width_pct=diagram_width)

    if prediction_text:
        st.markdown(
            f"<div style='text-align:center;font-size:1.2em'>{prediction_text}</div>",
            unsafe_allow_html=True
        )

# -----------------------------
# FALLBACK
# -----------------------------
else:
    st.error("Invalid mode. Use ?mode=input or ?mode=results.")
