import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# =========================
# LOAD DATA
# =========================
def load_data():
    df = pd.read_csv("data/credit_clean.csv")

    X = df.drop("Risk", axis=1)
    y = df["Risk"]

    return X, y

# =========================
# TRAIN MODELS
# =========================
def train_model():
    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Logistic Regression
    log_model = LogisticRegression(max_iter=1000)
    log_model.fit(X_train, y_train)
    log_proba = log_model.predict_proba(X_test)[:, 1]
    log_auc = roc_auc_score(y_test, log_proba)

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=150,
        max_depth=6,
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)

    print(f"Logistic AUC: {log_auc:.4f}")
    print(f"Random Forest AUC: {rf_auc:.4f}")

    # Best model
    best_model = rf_model if rf_auc > log_auc else log_model

    return best_model, {
        "log_auc": log_auc,
        "rf_auc": rf_auc,
        "X_test": X_test,
        "y_test": y_test
    }

# =========================
# PREDICT
# =========================
def predict(model, input_data):
    df = pd.DataFrame([input_data])

    prob = model.predict_proba(df)[0][1]
    score = int(prob * 100)

    decision = "REJECTED ❌" if prob > 0.5 else "APPROVED ✅"

    return {
        "probability": float(prob),
        "score": score,
        "decision": decision
    }

# =========================
# EXPLAINABILITY
# =========================
def explain_prediction(model, input_data):
    df = pd.DataFrame([input_data])

    # Si el modelo no tiene feature_importances (ej. Logistic)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        importances = abs(model.coef_[0])

    feature_names = df.columns

    explanation = []

    for i, feature in enumerate(feature_names):
        value = df.iloc[0, i]
        importance = importances[i]
        impact = value * importance

        explanation.append({
            "feature": feature,
            "impact": impact
        })

    explanation = sorted(explanation, key=lambda x: abs(x["impact"]), reverse=True)

    return explanation[:5]

# =========================
# TEST
# =========================
if __name__ == "__main__":
    model, metrics = train_model()

    sample = {
        "Age": 35,
        "Sex": 1,
        "Job": 2,
        "Housing": 1,
        "SavingAccounts": 1,
        "CheckingAccount": 1,
        "CreditAmount": 5000,
        "Duration": 24,
        "Purpose": 0
    }

    result = predict(model, sample)
    explanation = explain_prediction(model, sample)

    print("\n📊 RESULT:")
    print(result)

    print("\n🧠 EXPLANATION:")
    for e in explanation:
        print(e)