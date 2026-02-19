import streamlit as st
import joblib
import pandas as pd
import os
import train_model
import seaborn as sns
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_FILE = os.path.join(BASE_DIR, "vectorizer.pkl")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.csv")
ACCURACY_FILE = os.path.join(BASE_DIR, "accuracy_history.csv")
CONF_MATRIX_FILE = os.path.join(BASE_DIR, "confusion_matrix.csv")
METRICS_FILE = os.path.join(BASE_DIR, "metrics.csv")

st.set_page_config(page_title="Human-in-the-Loop Sentiment", layout="centered")
st.title("🤖 Human-in-the-Loop Sentiment System")

# ---------------- LOAD MODEL ----------------
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
else:
    st.warning("Model not found. Please train first.")
    st.stop()

# ---------------- SESSION STATE ----------------
if "prediction_done" not in st.session_state:
    st.session_state.prediction_done = False
if "last_text" not in st.session_state:
    st.session_state.last_text = ""
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = ""
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0

# ---------------- PREDICTION FORM ----------------
with st.form("prediction_form"):
    text_input = st.text_area("Enter Review Text")
    predict_btn = st.form_submit_button("Predict")

if predict_btn and text_input.strip():
    vector = vectorizer.transform([text_input])
    prediction = model.predict(vector)[0]
    confidence = max(model.predict_proba(vector)[0])

    st.session_state.prediction_done = True
    st.session_state.last_text = text_input
    st.session_state.last_prediction = prediction
    st.session_state.last_confidence = confidence

# ---------------- SHOW PREDICTION ----------------
if st.session_state.prediction_done:
    st.subheader("Prediction Result")
    st.write("Prediction:", st.session_state.last_prediction)
    st.write("Confidence:", f"{st.session_state.last_confidence*100:.2f}%")

    # ---------------- FEEDBACK FORM ----------------
    with st.form("feedback_form"):
        correct_label = st.radio(
            "Select Correct Label:",
            ("positive", "negative")
        )
        feedback_btn = st.form_submit_button("Submit Feedback")

    if feedback_btn:
        feedback_data = pd.DataFrame({
            "text": [st.session_state.last_text],
            "label": [correct_label]
        })
        if os.path.exists(FEEDBACK_FILE):
            feedback_data.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False)
        else:
            feedback_data.to_csv(FEEDBACK_FILE, mode="w", header=True, index=False)

        st.success("✅ Feedback Saved Successfully!")
        st.session_state.prediction_done = False  # Reset after feedback

st.divider()

# ---------------- RETRAIN MODEL ----------------
if st.button("🔄 Retrain Model"):
    st.info("Training model... This may take a few seconds.")
    acc = train_model.train_model()
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    st.success(f"Model Retrained! New Accuracy: {acc:.2f}%")

st.divider()

# ---------------- ACCURACY OVER TIME ----------------
if os.path.exists(ACCURACY_FILE):
    st.subheader("📈 Accuracy Over Time")
    acc_data = pd.read_csv(ACCURACY_FILE)
    if not acc_data.empty:
        st.line_chart(acc_data)
    else:
        st.info("No accuracy history yet.")

# ---------------- CONFUSION MATRIX ----------------
if os.path.exists(CONF_MATRIX_FILE):
    st.subheader("📊 Confusion Matrix")
    cm_data = pd.read_csv(CONF_MATRIX_FILE, index_col=0)
    if not cm_data.empty:
        fig, ax = plt.subplots()
        sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)
    else:
        st.info("No confusion matrix available.")

# ---------------- METRICS ----------------
if os.path.exists(METRICS_FILE):
    st.subheader("📊 Model Metrics")
    metrics = pd.read_csv(METRICS_FILE)
    if not metrics.empty:
        st.write(f"**Accuracy:** {metrics['accuracy'][0]:.2f}%")
        st.write(f"**Precision (Positive class):** {metrics['precision'][0]:.2f}%")
        st.write(f"**Recall (Positive class):** {metrics['recall'][0]:.2f}%")
        st.write(f"**F1-Score (Positive class):** {metrics['f1_score'][0]:.2f}%")
    else:
        st.info("Metrics file is empty.")
