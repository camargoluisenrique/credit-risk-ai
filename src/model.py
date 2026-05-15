import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "credit_clean.csv")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "credit_model.pkl")

os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

model_input_columns = None


def train_model():
    global model_input_columns

    df = pd.read_csv(DATA_PATH)

    y = df["Risk"]
    X = df.drop("Risk", axis=1)

    model_input_columns = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if y.nunique() == 2 else None,
    )

    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)

    log_auc = roc_auc_score(y_test, log_model.predict_proba(X_test)[:, 1])
    rf_auc = roc_auc_score(y_test, rf_model.predict_proba(X_test)[:, 1])

    best_model = rf_model if rf_auc >= log_auc else log_model

    joblib.dump((best_model, model_input_columns), MODEL_PATH)

    metrics = {
        "log_auc": log_auc,
        "rf_auc": rf_auc,
        "X_test": X_test,
        "y_test": y_test,
    }

    return best_model, metrics


def load_saved_model():
    global model_input_columns

    if not os.path.exists(MODEL_PATH):
        return None

    model, model_input_columns = joblib.load(MODEL_PATH)
    return model


def calculate_credit_score(probability):
    score = int(850 - (probability * 550))
    return max(300, min(850, score))


def credit_decision(probability):
    if probability > 0.70:
        return "REJECT"
    if probability > 0.40:
        return "REVIEW"
    return "APPROVE"


def prepare_input(input_data):
    global model_input_columns

    df = pd.DataFrame([input_data])

    for column in model_input_columns:
        if column not in df.columns:
            df[column] = 0

    return df[model_input_columns]


def predict(model, input_data):
    prepared_data = prepare_input(input_data)

    probability = model.predict_proba(prepared_data)[0][1]
    score = calculate_credit_score(probability)
    decision = credit_decision(probability)

    return {
        "probability": float(probability),
        "score": score,
        "decision": decision,
    }


def explain_prediction(model, input_data):
    prepared_data = prepare_input(input_data)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])
    else:
        return []

    explanation = []

    for index, column in enumerate(prepared_data.columns):
        value = prepared_data.iloc[0, index]
        impact = value * importances[index]

        explanation.append(
            {
                "feature": column,
                "impact": float(impact),
            }
        )

    explanation = sorted(
        explanation,
        key=lambda item: abs(item["impact"]),
        reverse=True,
    )

    return explanation[:5]