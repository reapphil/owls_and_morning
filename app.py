import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from filelock import FileLock

import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt

DATA_FILE = "responses.csv"
LOCK_FILE = DATA_FILE + ".lock"

st.set_page_config(page_title="AI Decision Boundary Demo", page_icon="🧠", layout="wide")
st.title("🧠 Decision Boundaries — Live Audience Demo")
st.write(    "Collect answers from your audience, train a simple classifier in real time, and **watch the decision boundary appear**.")

# -----------------------------
# Helpers for persistence
# -----------------------------

def init_store():
    if not os.path.exists(DATA_FILE):
        # Create empty store with all potential columns
        base_cols = [
            "timestamp", "session_id", "name", "scenario",
            # Morning vs Night
            "wake_time", "bed_time", "coffee_cups", "morning_energy", "label_morning_night",
            # Sweet vs Savory
            "sweet_to_savory", "desserts_per_week", "breakfast_pref", "label_sweet_savory",
            # Introvert vs Extrovert
            "party_vs_quiet", "conversations_yesterday", "drained_after_meetings", "label_social",
            # Early vs Late Adopter
            "update_speed", "smart_devices", "try_new_apps", "label_adopter",
        ]
        pd.DataFrame(columns=base_cols).to_csv(DATA_FILE, index=False)


def append_row(row: dict):
    lock = FileLock(LOCK_FILE)
    with lock:
        df = pd.read_csv(DATA_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)


def load_data():
    if not os.path.exists(DATA_FILE):
        init_store()
    try:
        df = pd.read_csv(DATA_FILE)
    except Exception:
        df = pd.DataFrame()
    return df


# -----------------------------
# Scenario definitions
# -----------------------------
SCENARIOS = {
    "Morning vs Night": {
        "label_col": "label_morning_night",
        "label_map": {0: "Night Owl", 1: "Morning Person"},
        "features": [
            ("wake_time", "Wake-up time (0–23h)"),
            ("bed_time", "Bedtime (0–23h)"),
            ("coffee_cups", "Cups of coffee/tea per day"),
            ("morning_energy", "How energetic before noon? (1–10)"),
        ],
        "label_prompt": "How do you self-identify?",
        "label_choices": ["Night Owl", "Morning Person"],
    },
    "Sweet vs Savory": {
        "label_col": "label_sweet_savory",
        "label_map": {0: "Savory Lover", 1: "Sweet Tooth"},
        "features": [
            ("sweet_to_savory", "Snack preference (0=savory, 10=sweet)"),
            ("desserts_per_week", "Desserts per week"),
            ("breakfast_pref", "Breakfast: 0=eggs/cheese, 10=pastry"),
        ],
        "label_prompt": "Which do you prefer overall?",
        "label_choices": ["Savory", "Sweet"],
    },
    "Introvert vs Extrovert": {
        "label_col": "label_social",
        "label_map": {0: "Introvert-leaning", 1: "Extrovert-leaning"},
        "features": [
            ("party_vs_quiet", "Party (10) vs Quiet night (0)"),
            ("conversations_yesterday", "How many distinct conversations yesterday?"),
            ("drained_after_meetings", "How drained after group meetings? (0–10)"),
        ],
        "label_prompt": "Where do you net out?",
        "label_choices": ["Introvert-leaning", "Extrovert-leaning"],
    },
    "Early vs Late Adopter": {
        "label_col": "label_adopter",
        "label_map": {0: "Late Adopter", 1: "Early Adopter"},
        "features": [
            ("update_speed", "How fast do you install updates? (0=wait, 10=immediately)"),
            ("smart_devices", "How many smart devices at home?"),
            ("try_new_apps", "How often try new apps? (0–10)"),
        ],
        "label_prompt": "Do you consider yourself an early tech adopter?",
        "label_choices": ["No", "Yes"],
    },
}

# -----------------------------
# Sidebar Config
# -----------------------------
with st.sidebar:
    st.header("Configuration")
    scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()))
    clf_name = st.selectbox("Classifier", ["Logistic Regression", "k-NN (k=7)", "Decision Tree"], index=0)
    allow_names = st.checkbox("Collect first names (optional)", value=True)
    clear_btn = st.button("🧹 Reset demo data (local session)")
    if clear_btn:
        lock = FileLock(LOCK_FILE)
        with lock:
            init_store()  # recreates empty file
        st.success("Data reset. Start collecting anew.")

scenario = SCENARIOS[scenario_name]

# -----------------------------
# Audience Input Form
# -----------------------------
with st.form("audience_form", clear_on_submit=True):
    st.subheader("1) Audience: enter your answers")
    cols = st.columns(2)
    name = None
    if allow_names:
        name = cols[0].text_input("First name (optional)")
    session_id = st.session_state.get("_xsrf", "session")  # lightweight per-user id

    # Render scenario-specific inputs
    values = {}
    if scenario_name == "Morning vs Night":
        values["wake_time"] = cols[0].number_input("Wake-up time (0–23)", 0, 23, 7)
        values["bed_time"] = cols[1].number_input("Bedtime (0–23)", 0, 23, 23)
        values["coffee_cups"] = cols[0].number_input("Cups of coffee/tea per day", 0, 10, 1)
        values["morning_energy"] = cols[1].slider("How energetic before noon?", 0, 10, 6)
        label_choice = st.radio(scenario["label_prompt"], scenario["label_choices"], horizontal=True)
        label_val = 1 if label_choice in ("Morning Person",) else 0
    elif scenario_name == "Sweet vs Savory":
        values["sweet_to_savory"] = cols[0].slider("Snack preference (savory→sweet)", 0, 10, 5)
        values["desserts_per_week"] = cols[1].number_input("Desserts per week", 0, 21, 3)
        values["breakfast_pref"] = cols[0].slider("Breakfast (0=eggs, 10=pastry)", 0, 10, 6)
        label_choice = st.radio(scenario["label_prompt"], scenario["label_choices"], horizontal=True)
        label_val = 1 if label_choice == "Sweet" else 0
    elif scenario_name == "Introvert vs Extrovert":
        values["party_vs_quiet"] = cols[0].slider("Party (10) vs Quiet night (0)", 0, 10, 5)
        values["conversations_yesterday"] = cols[1].number_input("Conversations yesterday", 0, 60, 6)
        values["drained_after_meetings"] = cols[0].slider("How drained after meetings?", 0, 10, 5)
        label_choice = st.radio(scenario["label_prompt"], scenario["label_choices"], horizontal=True)
        label_val = 1 if label_choice.startswith("Extrovert") else 0
    else:  # Early vs Late Adopter
        values["update_speed"] = cols[0].slider("Install updates speed", 0, 10, 7)
        values["smart_devices"] = cols[1].number_input("# of smart devices", 0, 50, 5)
        values["try_new_apps"] = cols[0].slider("How often try new apps?", 0, 10, 6)
        label_choice = st.radio(scenario["label_prompt"], scenario["label_choices"], horizontal=True)
        label_val = 1 if label_choice == "Yes" else 0

    submitted = st.form_submit_button("Submit my answers ✅")

    if submitted:
        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "session_id": session_id,
            "name": name,
            "scenario": scenario_name,
        }
        row.update({k: values.get(k) for k, _ in scenario["features"]})
        row[scenario["label_col"]] = label_val
        # Fill missing columns to keep CSV rectangular
        for sname, sc in SCENARIOS.items():
            for f, _ in sc["features"]:
                row.setdefault(f, np.nan)
            row.setdefault(sc["label_col"], np.nan)
        append_row(row)
        st.success("Thanks! Your point will appear in the plot below.")

st.markdown("---")

# -----------------------------
# Visualization & Modeling
# -----------------------------
colL, colR = st.columns([2, 1])

with colL:
    st.subheader("2) Live scatter + decision boundary")

    df = load_data()
    df = df[df["scenario"] == scenario_name].copy()

    if df.empty or df[scenario["label_col"]].dropna().empty:
        st.info("No data yet — submit a few responses to get started.")
        st.stop()

    # Prepare dataset
    label_col = scenario["label_col"]
    label_series = df[label_col].dropna()
    X_df = df[[f for f, _ in scenario["features"]]]
    y = label_series.astype(int).values
    X = X_df.values.astype(float)

    # Drop rows with NaNs
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X, y = X[mask], y[mask]
    names = df.loc[mask, "name"].fillna("").values

    if len(y) < 6:
        st.warning("Collect ~6+ responses for a clearer boundary.")

    # Choose top-2 features via mutual information
    try:
        mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    except Exception:
        mi = np.zeros(X.shape[1])
    feat_order = np.argsort(mi)[::-1]
    i1, i2 = feat_order[:2] if X.shape[1] >= 2 else (0, 0)

    feat_names = [scenario["features"][i1][1], scenario["features"][i2][1]]
    used_cols = [scenario["features"][i1][0], scenario["features"][i2][0]]

    X2 = X[:, [i1, i2]]

    # Model selection
    if clf_name.startswith("Logistic"):
        clf = LogisticRegression()
    elif clf_name.startswith("k-NN"):
        clf = KNeighborsClassifier(n_neighbors=7)
    else:
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)

    scaler = StandardScaler()
    X2s = scaler.fit_transform(X2)
    clf.fit(X2s, y)

    # Grid for decision boundary
    x_min, x_max = X2[:, 0].min() - 0.5, X2[:, 0].max() + 0.5
    y_min, y_max = X2[:, 1].min() - 0.5, X2[:, 1].max() + 0.5
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 200),
        np.linspace(y_min, y_max, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = clf.predict_proba(scaler.transform(grid))[:, 1].reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    cs = ax.contourf(xx, yy, Z, levels=30, alpha=0.3)
    # Plot points
    for cls, marker in [(0, "o"), (1, "^")]:
        idx = (y == cls)
        ax.scatter(X2[idx, 0], X2[idx, 1], marker=marker, edgecolor="k", s=80, label=scenario["label_map"][cls])
    # Optional annotations
    for (x, ypt, nm) in zip(X2[:, 0], X2[:, 1], names):
        if nm:
            ax.annotate(nm, (x, ypt), xytext=(3, 3), textcoords="offset points", fontsize=8)

    ax.set_xlabel(feat_names[0])
    ax.set_ylabel(feat_names[1])
    ax.set_title(f"{scenario_name} — {clf_name}")
    ax.legend(loc="best")
    st.pyplot(fig, clear_figure=True)

with colR:
    st.subheader("3) Quick stats")
    st.caption("(On the top-2 features)")

    # Train/test split (simple) for quick accuracy read
    if len(y) >= 10:
        # simple holdout: last 20% as test
        n = len(y)
        split = max(1, int(0.8 * n))
        Xtr, Xte = X2s[:split], X2s[split:]
        ytr, yte = y[:split], y[split:]
        clf2 = clf.__class__(**getattr(clf, 'get_params', lambda: {})()) if hasattr(clf, 'get_params') else clf
        clf2.fit(Xtr, ytr)
        ypred = clf2.predict(Xte)
        acc = accuracy_score(yte, ypred)
        st.metric("Holdout accuracy", f"{acc*100:.1f}%")
    else:
        st.write("Need ~10+ responses for a quick accuracy estimate.")

    st.write("**Top features by mutual information:**")
    rank_df = pd.DataFrame({
        "Feature": [label for _, label in scenario["features"]],
        "MI": mi,
    }).sort_values("MI", ascending=False)
    st.dataframe(rank_df, use_container_width=True, hide_index=True)

    st.download_button(
        label="⬇️ Download raw responses (CSV)",
        data=pd.read_csv(DATA_FILE).to_csv(index=False).encode("utf-8"),
        file_name="responses_export.csv",
        mime="text/csv",
    )

st.markdown("""---**Privacy note:** This demo purposely avoids sensitive traits (e.g., age, gender, health). All questions are light-weight preferences.""")
