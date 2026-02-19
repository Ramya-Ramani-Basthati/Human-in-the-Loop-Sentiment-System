import pandas as pd
import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(BASE_DIR, "data.csv")
FEEDBACK_FILE = os.path.join(BASE_DIR, "feedback.csv")
MODEL_FILE = os.path.join(BASE_DIR, "model.pkl")
VECTORIZER_FILE = os.path.join(BASE_DIR, "vectorizer.pkl")
ACCURACY_FILE = os.path.join(BASE_DIR, "accuracy_history.csv")
CONF_MATRIX_FILE = os.path.join(BASE_DIR, "confusion_matrix.csv")
METRICS_FILE = os.path.join(BASE_DIR, "metrics.csv")  # New file for precision, recall, f1

def train_model(verbose=False):
    # Check main dataset
    if not os.path.exists(DATA_FILE):
        if verbose:
            print("data.csv not found!")
        return 0

    if verbose:
        print("Loading main dataset...")
    data = pd.read_csv(DATA_FILE)

    if verbose:
        print("Main dataset size:", len(data))

    # Merge feedback if exists
    if os.path.exists(FEEDBACK_FILE):
        feedback = pd.read_csv(FEEDBACK_FILE)
        if verbose:
            print("Feedback file found. Merging feedback...")
            print("Feedback size:", len(feedback))
        data = pd.concat([data, feedback], ignore_index=True)

    data.dropna(inplace=True)
    data["label"] = data["label"].str.lower().str.strip()

    X = data["text"]
    y = data["label"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000
    )

    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=500
    )

    model.fit(X_train_vec, y_train)
    y_pred = model.predict(X_test_vec)

    # ----- Metrics -----
    accuracy = accuracy_score(y_test, y_pred) * 100
    precision = precision_score(y_test, y_pred, pos_label="positive") * 100
    recall = recall_score(y_test, y_pred, pos_label="positive") * 100
    f1 = f1_score(y_test, y_pred, pos_label="positive") * 100

    # Save model and vectorizer
    joblib.dump(model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)

    # Save accuracy history
    acc_df = pd.DataFrame({"accuracy": [accuracy]})
    if os.path.exists(ACCURACY_FILE):
        acc_df.to_csv(ACCURACY_FILE, mode="a", header=False, index=False)
    else:
        acc_df.to_csv(ACCURACY_FILE, mode="w", header=True, index=False)

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=["positive", "negative"])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual Positive", "Actual Negative"],
        columns=["Predicted Positive", "Predicted Negative"]
    )
    cm_df.to_csv(CONF_MATRIX_FILE)

    # Save precision, recall, f1
    metrics_df = pd.DataFrame({
        "accuracy": [accuracy],
        "precision": [precision],
        "recall": [recall],
        "f1_score": [f1]
    })
    metrics_df.to_csv(METRICS_FILE, index=False)

    if verbose:
        print("Model saved successfully.")
        print("Confusion matrix saved.")
        print(f"Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1-Score: {f1:.2f}%")

    return round(accuracy, 2)

# ---------------- RUN AS SCRIPT ----------------
if __name__ == "__main__":
    acc = train_model(verbose=True)
    print("Model Retrained Successfully!")
    print("Accuracy:", acc, "%")
