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
st.set_page_config(page_title="Morning vs. Night AI Demo", page_icon="🧠", layout="wide")

DATA_FILE = "responses.csv"
LOCK_FILE = DATA_FILE + ".lock"

# -----------------------------
# Data utilities
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

    morning = pd.DataFrame({
        "timestamp": [datetime.now(timezone.utc).isoformat()] * n_half,
        "wake_time": np.clip(np.random.normal(0.35, 0.15, n_half), 0, 1),
        "bed_time": np.clip(np.random.normal(0.4, 0.15, n_half), 0, 1),
        "coffee": np.clip(np.random.normal(0.45, 0.2, n_half), 0, 1),
        "energy": np.clip(np.random.normal(0.7, 0.2, n_half), 0, 1),
        "label": [1] * n_half,
    })

    night = pd.DataFrame({
        "timestamp": [datetime.now(timezone.utc).isoformat()] * n_half,
        "wake_time": np.clip(np.random.normal(0.65, 0.15, n_half), 0, 1),
        "bed_time": np.clip(np.random.normal(0.65, 0.15, n_half), 0, 1),
        "coffee": np.clip(np.random.normal(0.6, 0.2, n_half), 0, 1),
        "energy": np.clip(np.random.normal(0.45, 0.2, n_half), 0, 1),
        "label": [0] * n_half,
    })

    df = pd.concat([morning, night], ignore_index=True)

    df["wake_time"] = np.clip(
        df["wake_time"] * 0.7 + df["bed_time"] * 0.3 + np.random.normal(0, 0.05, n),
        0, 1
    )

    flip_mask = np.random.rand(n) < 0.05
    df.loc[flip_mask, "label"] = 1 - df.loc[flip_mask, "label"]

    return df.sample(frac=1, random_state=99).reset_index(drop=True)

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
    coffee = st.slider("Coffee consumption (no coffee ⟶ a lot of coffee)", 0.0, 1.0, 0.5)
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
    st.markdown(
        "<h2 style='text-align:center'>📊 Morning (Blue) vs. Night (Red) — Results "
        "</h2>",
        unsafe_allow_html=True
    )

    if st.button("🗑️ Clear all responses"):
        if os.path.exists(DATA_FILE):
            os.remove(DATA_FILE)
        st.success("All responses deleted. Synthetic data will repopulate on next load.")
        st.stop()

    df = load_data()
    df = df.dropna(subset=["wake_time", "bed_time", "label"])
    X = df[["wake_time", "bed_time"]].astype(float).values
    y = df["label"].astype(int).values

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
        st.sidebar.subheader("🧠 Neural Network Settings")
        num_layers = st.sidebar.slider("Hidden layers", 1, 4, 2)
        neurons = st.sidebar.slider("Neurons per layer", 2, 20, 8)
        activation = st.sidebar.selectbox("Activation", ["relu", "tanh", "logistic"])
        alpha = st.sidebar.slider("Regularization (alpha)", 0.0001, 0.05, 0.001, step=0.0005)
        lr = st.sidebar.slider("Learning rate", 0.0001, 0.1, 0.01, step=0.0005)
        iters = st.sidebar.slider("Iterations", 200, 5000, 2000, step=100)

        hidden_layers = tuple([neurons] * num_layers)
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            alpha=alpha,
            learning_rate_init=lr,
            max_iter=iters,
            random_state=42
        )

    # Train model
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model.fit(Xs, y)

    # Decision boundary
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )
    Z = model.predict_proba(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))[:, 1].reshape(xx.shape)

    # Optional random point prediction
    show_prediction = st.button("🎲 Generate Random Person")

    random_point = None
    prediction_text = None
    if show_prediction:
        random_point = np.random.rand(1, 2)  # (wake_time, bed_time)
        scaled_pt = scaler.transform(random_point)
        pred = model.predict(scaled_pt)[0]
        pred_label = "🌅 Morning Person" if pred == 1 else "🌙 Night Owl"
        prediction_text = f"Model prediction for random person: **{pred_label}**"

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    cs = ax.contourf(xx, yy, Z, levels=30, cmap="coolwarm", alpha=0.25)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="bwr", edgecolor="k", s=70)

    if random_point is not None:
        ax.scatter(random_point[:, 0], random_point[:, 1],
                   s=150, color="limegreen", edgecolor="black", marker="o", label="Random Person")
        ax.legend(loc="upper left")

    ax.set_xlabel("Wake-up time (early ⟶ late)")
    ax.set_ylabel("Bedtime (early ⟶ late)")
    ax.set_title(f"Decision Boundary — {model_name}", fontsize=12)
    render_matplotlib(fig, width_pct=50)

    if prediction_text:
        st.markdown(f"<p style='text-align:center;font-size:1.1em;'>{prediction_text}</p>", unsafe_allow_html=True)

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
    st.error("Unknown mode. Use ?mode=input or ?mode=results in the URL.")
