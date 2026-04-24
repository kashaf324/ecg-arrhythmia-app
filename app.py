import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="ECG Classifier", layout="wide")

# =========================
# TITLE
# =========================
st.title("🫀 ECG Arrhythmia Detection Dashboard")
st.markdown("### AI-powered classification of heart rhythms")

# =========================
# FILE UPLOAD
# =========================
uploaded_file = st.file_uploader("Upload ECG CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "")

    st.subheader("📂 Input Data")
    st.dataframe(df.head())

    # Save actual labels
    actual = df["type"] if "type" in df.columns else None

    # Drop non-features
    if "record" in df.columns:
        df = df.drop(columns=["record"])
    if "type" in df.columns:
        df = df.drop(columns=["type"])

    # =========================
    # PREDICT
    # =========================
    X = scaler.transform(df)
    preds = model.predict(X)
    labels = label_encoder.inverse_transform(preds)

    df["Prediction"] = labels

    # =========================
    # ABNORMAL DETECTION
    # =========================
    abnormal_df = df[df["Prediction"] != "N"]

    # =========================
    # DASHBOARD METRICS
    # =========================
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Samples", len(df))
    col2.metric("Normal Beats", (df["Prediction"] == "N").sum())
    col3.metric("Abnormal Beats", len(abnormal_df))

    # =========================
    # ALERT SECTION
    # =========================
    if len(abnormal_df) > 0:
        st.error(f"⚠️ {len(abnormal_df)} abnormal beats detected!")
    else:
        st.success("✅ No abnormal beats detected")

    # =========================
    # SHOW ABNORMAL ONLY
    # =========================
    st.subheader("🚨 Abnormal Beats (First 20)")
    st.dataframe(abnormal_df.head(20))

    # =========================
    # DISTRIBUTION
    # =========================
    st.subheader("📊 Class Distribution")
    st.bar_chart(df["Prediction"].value_counts())

    # =========================
    # OPTIONAL COMPARISON
    # =========================
    if actual is not None:
        st.subheader("🔍 Actual vs Predicted")
        comparison = pd.DataFrame({
            "Actual": actual,
            "Predicted": labels
        })
        st.dataframe(comparison.head(20))