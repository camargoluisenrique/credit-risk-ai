import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# =========================
# PATHS
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "credit_model.pkl")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# =========================
# GLOBAL
# =========================
model_input_columns = None

# =========================
# TRAIN
# =========================
def train_and_save_model():
    global model_input_columns

    df = pd.read_csv(DATA_PATH)

    y = df["Risk"]
    X = df.drop("Risk", axis=1)

    model_input_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Logistic (baseline interpretable)
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    # Random Forest (non-linear)
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    # evaluación
    log_auc = roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1])
    rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

    print(f"Logistic AUC: {log_auc:.4f}")
    print(f"Random Forest AUC: {rf_auc:.4f}")

    best_model = rf_model if rf_auc > log_auc else log_model

    # guardar
    joblib.dump((best_model, model_input_columns), MODEL_PATH)

    return best_model, {
        "log_auc": log_auc,
        "rf_auc": rf_auc,
        "X_test": X_test,
        "y_test": y_test
    }

# =========================
# LOAD
# =========================
def load_model():
    global model_input_columns

    if os.path.exists(MODEL_PATH):
        model, model_input_columns = joblib.load(MODEL_PATH)
        return model

    model, _ = train_and_save_model()
    return model

# =========================
# SCORE (tipo banca)
# =========================
def calculate_credit_score(prob):
    # escala tipo FICO (300–850)
    score = int(850 - (prob * 550))
    return max(300, min(850, score))

# =========================
# DECISION LOGIC (REAL)
# =========================
def credit_decision(prob):

    if prob > 0.7:
        return "REJECT"
    elif prob > 0.4:
        return "REVIEW"
    else:
        return "APPROVE"

# =========================
# PREDICT
# =========================
def predict(model, input_data):
    global model_input_columns

    df = pd.DataFrame([input_data])

    for col in model_input_columns:
        if col not in df.columns:
            df[col] = 0

    df = df[model_input_columns]

    prob = model.predict_proba(df)[0][1]

    score = calculate_credit_score(prob)
    decision = credit_decision(prob)

    return {
        "probability": float(prob),
        "score": score,
        "decision": decision
    }

# =========================
# EXPLAINABILITY (MEJORADA)
# =========================
def explain_prediction(model, input_data):
    df = pd.DataFrame([input_data])

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = abs(model.coef_[0])

    explanation = []

    for i, col in enumerate(df.columns):
        impact = df.iloc[0, i] * importances[i]

        explanation.append({
            "feature": col,
            "impact": float(impact)
        })

    explanation = sorted(
        explanation,
        key=lambda x: abs(x["impact"]),
        reverse=True
    )

    return explanation[:5]