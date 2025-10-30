import os
from datetime import datetime, timezone
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
st.write("Collect answers from your audience, train a simple classifier in real time, and **watch the decision boundary appear**.")

def init_store():
    if not os.path.exists(DATA_FILE):
        base_cols = [
            "timestamp", "session_id", "name", "scenario",
            "wake_time", "bed_time", "coffee_cups", "morning_energy", "label_morning_night",
            "sweet_to_savory", "desserts_per_week", "breakfast_pref", "label_sweet_savory",
            "party_vs_quiet", "conversations_yesterday", "drained_after_meetings", "label_social",
            "update_speed", "smart_devices", "try_new_apps", "label_adopter",
        ]
        pd.DataFrame(columns=base_cols).to_csv(DATA_FILE, index=False)

def append_row(row):
    lock = FileLock(LOCK_FILE)
    with lock:
        df = pd.read_csv(DATA_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)

def load_data():
    if not os.path.exists(DATA_FILE):
        init_store()
    return pd.read_csv(DATA_FILE)

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
}

with st.sidebar:
    scenario_name = st.selectbox("Scenario", list(SCENARIOS.keys()))
    clf_name = st.selectbox("Classifier", ["Logistic Regression", "k-NN (k=7)", "Decision Tree"])
    clear_btn = st.button("🧹 Reset demo data")
    if clear_btn:
        init_store()
        st.success("Data reset.")

scenario = SCENARIOS[scenario_name]

with st.form("audience_form", clear_on_submit=True):
    st.subheader("Submit your answers")
    cols = st.columns(2)
    name = cols[0].text_input("First name (optional)")
    session_id = st.session_state.get("_xsrf", "session")

    if scenario_name == "Morning vs Night":
        wake_time = cols[0].number_input("Wake-up time (0–23)", 0, 23, 7)
        bed_time = cols[1].number_input("Bedtime (0–23)", 0, 23, 23)
        coffee_cups = cols[0].number_input("Cups of coffee/tea per day", 0, 10, 1)
        morning_energy = cols[1].slider("How energetic before noon?", 0, 10, 6)
        label_choice = st.radio(scenario["label_prompt"], scenario["label_choices"], horizontal=True)
        label_val = 1 if label_choice == "Morning Person" else 0
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "name": name,
            "scenario": scenario_name,
            "wake_time": wake_time,
            "bed_time": bed_time,
            "coffee_cups": coffee_cups,
            "morning_energy": morning_energy,
            "label_morning_night": label_val,
        }
    else:
        sweet_to_savory = cols[0].slider("Snack preference (0=savory,10=sweet)", 0, 10, 5)
        desserts_per_week = cols[1].number_input("Desserts per week", 0, 21, 3)
        breakfast_pref = cols[0].slider("Breakfast (0=eggs,10=pastry)", 0, 10, 6)
        label_choice = st.radio(scenario["label_prompt"], scenario["label_choices"], horizontal=True)
        label_val = 1 if label_choice == "Sweet" else 0
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "name": name,
            "scenario": scenario_name,
            "sweet_to_savory": sweet_to_savory,
            "desserts_per_week": desserts_per_week,
            "breakfast_pref": breakfast_pref,
            "label_sweet_savory": label_val,
        }

    submitted = st.form_submit_button("Submit ✅")
    if submitted:
        append_row(row)
        st.success("Thanks! Your point will appear below.")

st.markdown("---")

colL, colR = st.columns([2, 1])

with colL:
    df = load_data()
    df = df[df["scenario"] == scenario_name].dropna(subset=[scenario["label_col"]])
    if df.empty:
        st.info("No data yet — submit a few responses.")
        st.stop()

    X = df[[f for f, _ in scenario["features"]]].astype(float).values
    y = df[scenario["label_col"]].astype(int).values

    if len(np.unique(y)) < 2:
        st.warning("Need at least two classes to train the model.")
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.scatter(X[:, 0], X[:, 1], s=80, edgecolor='k')
        st.pyplot(fig)
        st.stop()

    mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)
    order = np.argsort(mi)[::-1]
    i1, i2 = order[:2]
    X2 = X[:, [i1, i2]]

    if clf_name.startswith("Logistic"):
        clf = LogisticRegression()
    elif clf_name.startswith("k-NN"):
        k_val = min(7, len(y))
        clf = KNeighborsClassifier(n_neighbors=max(1, k_val))
    else:
        clf = DecisionTreeClassifier(max_depth=4, random_state=42)

    scaler = StandardScaler()
    X2s = scaler.fit_transform(X2)
    clf.fit(X2s, y)

    x_min, x_max = X2[:, 0].min()-0.5, X2[:, 0].max()+0.5
    y_min, y_max = X2[:, 1].min()-0.5, X2[:, 1].max()+0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(scaler.transform(grid))
        if proba.shape[1] == 1:
            Z = np.zeros_like(xx)
        else:
            pos_idx = int(np.where(clf.classes_ == 1)[0][0])
            Z = proba[:, pos_idx].reshape(xx.shape)
    else:
        preds = clf.predict(scaler.transform(grid))
        Z = preds.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, Z, levels=30, alpha=0.3)
    for cls, marker in [(0, 'o'), (1, '^')]:
        idx = y == cls
        ax.scatter(X2[idx, 0], X2[idx, 1], marker=marker, edgecolor='k', s=80)
    ax.set_xlabel(scenario["features"][i1][1])
    ax.set_ylabel(scenario["features"][i2][1])
    ax.set_title(f"{scenario_name} — {clf_name}")
    st.pyplot(fig)

with colR:
    st.subheader("Quick stats")
    if len(y) >= 10:
        n = len(y)
        split = int(0.8 * n)
        Xtr, Xte = X2s[:split], X2s[split:]
        ytr, yte = y[:split], y[split:]
        clf2 = clf.__class__(**getattr(clf, 'get_params', lambda: {})())
        clf2.fit(Xtr, ytr)
        acc = accuracy_score(yte, clf2.predict(Xte))
        st.metric("Holdout accuracy", f"{acc*100:.1f}%")
    else:
        st.write("Need ~10+ responses for accuracy estimate.")