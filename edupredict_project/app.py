import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)

st.set_page_config(
    page_title="EduPredict Studio – ANN Student Performance",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 EduPredict: Student Performance Prediction Using Artificial Neural Networks")
st.markdown("### MBA Data Science & AI – Interactive Cohort Synthesis, Zero-Leakage Preprocessing & ANN Diagnostics")

@st.cache_data
def generate_synthetic_students(n_samples=1000, passing_cutoff=40, seed=2026):
    rng = np.random.default_rng(seed)
    student_ids = [f"STU-{i:04d}" for i in range(1, n_samples + 1)]
    age = rng.integers(16, 26, size=n_samples)
    gender = rng.choice(["Female", "Male"], size=n_samples, p=[0.51, 0.49])
    diligence = np.clip(rng.normal(0, 1, size=n_samples), -2.5, 2.5)

    attendance = np.round(np.clip(74 + diligence * 11 + rng.normal(0, 7, size=n_samples), 40, 100), 1)
    study_hours = np.round(np.clip(4.6 + diligence * 1.6 + rng.normal(0, 1.3, size=n_samples), 0, 10), 1)
    previous_score = np.round(np.clip(62 + diligence * 13 + rng.normal(0, 9, size=n_samples), 0, 100), 1)
    assignment_score = np.round(np.clip(66 + diligence * 14 + rng.normal(0, 8, size=n_samples), 0, 100), 1)
    participation = np.round(np.clip(60 + diligence * 15 + rng.normal(0, 11, size=n_samples), 0, 100), 1)
    sleep_hours = np.round(np.clip(6.8 + rng.normal(0, 1.3, size=n_samples), 3, 10), 1)
    internet_access = rng.choice(["Yes", "No"], size=n_samples, p=[0.82, 0.18])
    extracurricular = rng.choice(["Yes", "No"], size=n_samples, p=[0.55, 0.45])

    sleep_bonus = np.clip(4.5 - np.abs(sleep_hours - 7.5) * 1.4, -2.0, 4.5)
    raw_final = (
        -18.0
        + 0.28 * attendance
        + 2.45 * study_hours
        + 0.24 * previous_score
        + 0.22 * assignment_score
        + 0.10 * participation
        + sleep_bonus
        + np.where(internet_access == "Yes", 2.5, 0.0)
        + np.where(extracurricular == "Yes", 1.5, 0.0)
        + rng.normal(0, 6.2, size=n_samples)
    )
    final_score = np.round(np.clip(raw_final, 0, 100), 1)
    performance = (final_score >= passing_cutoff).astype(int)

    return pd.DataFrame({
        "Student_ID": student_ids,
        "Age": age,
        "Gender": gender,
        "Attendance": attendance,
        "Study_Hours": study_hours,
        "Previous_Score": previous_score,
        "Assignment_Score": assignment_score,
        "Participation": participation,
        "Sleep_Hours": sleep_hours,
        "Internet_Access": internet_access,
        "Extracurricular": extracurricular,
        "Final_Score": final_score,
        "Performance": performance
    })

st.sidebar.header("1. Student Data Source")
data_mode = st.sidebar.radio("Cohort Source", ["Generate Synthetic Cohort", "Upload Custom CSV"])

if data_mode == "Generate Synthetic Cohort":
    n_samples = st.sidebar.selectbox("Dataset Size", [100, 500, 1000, 2000, 5000], index=2)
    passing_cutoff = st.sidebar.slider("Passing Score Cutoff", 25, 75, 40)
    seed = st.sidebar.number_input("Random Seed", value=2026, step=1)
    df = generate_synthetic_students(n_samples, passing_cutoff, seed)
    target_col = "Performance"
    default_features = [c for c in df.columns if c not in ("Student_ID", "Final_Score", "Performance")]
else:
    uploaded = st.sidebar.file_uploader("Upload Student CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV file or switch to 'Generate Synthetic Cohort'.")
        st.stop()
    df = pd.read_csv(uploaded)
    target_col = st.sidebar.selectbox("Target Column", df.columns, index=len(df.columns) - 1)
    default_features = [c for c in df.columns if c not in ("Student_ID", "Final_Score", target_col)]

selected_features = st.sidebar.multiselect("Input Features (X)", [c for c in df.columns if c != target_col], default=default_features)
if not selected_features:
    st.error("Please select at least one input feature.")
    st.stop()

st.sidebar.header("2. ANN Architecture & Hyperparameters")
hidden1 = st.sidebar.slider("Hidden Layer 1 Neurons (ReLU)", 4, 64, 16, step=4)
hidden2 = st.sidebar.selectbox("Second Hidden Layer", [0, 8, 12, 16], index=0)
epochs = st.sidebar.slider("Epochs (Max Iterations)", 10, 150, 50, step=10)
batch_size = st.sidebar.selectbox("Mini-Batch Size", [16, 32, 64], index=1)
lr = st.sidebar.selectbox("Adam Learning Rate", [0.0005, 0.001, 0.005, 0.01], index=1)
test_ratio = st.sidebar.selectbox("Test Split Ratio", [0.15, 0.20, 0.30], index=1)
threshold = st.sidebar.slider("Decision Threshold (tau)", 0.10, 0.90, 0.50, step=0.02)

X = df[selected_features]
y = df[target_col].astype(int).values

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in selected_features if c not in num_cols]

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([("impute", SimpleImputer(strategy="mean")), ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=42, stratify=y)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

layers = (hidden1, hidden2) if hidden2 > 0 else (hidden1,)
mlp = MLPClassifier(hidden_layer_sizes=layers, activation="relu", solver="adam", learning_rate_init=lr, batch_size=batch_size, max_iter=epochs, random_state=42)
mlp.fit(X_train_proc, y_train)

y_prob = mlp.predict_proba(X_test_proc)[:, 1]
y_pred = (y_prob >= threshold).astype(int)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_test, y_pred)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Held-Out Test Accuracy", f"{acc*100:.1f}%")
c2.metric("Test Precision", f"{prec:.3f}")
c3.metric("Test Recall", f"{rec:.3f}")
c4.metric("Test F1-Score", f"{f1:.3f}")
c5.metric("Test ROC-AUC", f"{roc_auc:.3f}")

st.divider()
col_l, col_r = st.columns(2)
with col_l:
    st.subheader("Training Loss Convergence (Binary Cross-Entropy)")
    fig_loss = px.line(y=mlp.loss_curve_, labels={"index": "Epoch", "y": "BCE Loss"})
    st.plotly_chart(fig_loss, use_container_width=True)
with col_r:
    st.subheader("ROC Curve (Held-Out Test Set)")
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ANN ROC (AUC={roc_auc:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random Baseline", line=dict(dash="dash")))
    st.plotly_chart(fig_roc, use_container_width=True)

st.subheader("Active Student Cohort Preview")
st.dataframe(df.head(15), use_container_width=True)
