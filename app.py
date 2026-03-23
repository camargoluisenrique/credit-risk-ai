import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from model import train_model, predict, explain_prediction

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Credit Risk AI", layout="centered")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return train_model()

model, metrics = load_model()

# =========================
# TITLE
# =========================
st.title("💳 AI Credit Risk Scoring")

st.markdown("""
This application simulates a real-world credit risk system used in fintech and banking.  
It predicts the probability of default and supports loan approval decisions.
""")

# =========================
# INPUT
# =========================
st.subheader("📋 Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 75, 30)

    job_map = {"Unskilled": 0, "Skilled": 1, "Highly Skilled": 2, "Executive": 3}
    job = job_map[st.selectbox("Job Level", list(job_map.keys()))]

    housing_map = {"Rent": 0, "Own": 1, "Free": 2}
    housing = housing_map[st.selectbox("Housing", list(housing_map.keys()))]

with col2:
    credit_amount = st.number_input("Credit Amount", 100, 20000, 5000)
    duration = st.slider("Duration (months)", 6, 72, 24)

    purpose_map = {
        "Car": 0, "Furniture": 1, "Education": 2,
        "Business": 3, "Appliances": 4,
        "Repairs": 5, "Vacation": 6, "Other": 7
    }
    purpose = purpose_map[st.selectbox("Purpose", list(purpose_map.keys()))]

saving_map = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3}
saving = saving_map[st.selectbox("Saving Accounts", list(saving_map.keys()))]

checking_map = {"Low": 0, "Moderate": 1, "High": 2, "Very High": 3}
checking = checking_map[st.selectbox("Checking Account", list(checking_map.keys()))]

sex_map = {"Male": 1, "Female": 0}
sex = sex_map[st.selectbox("Sex", list(sex_map.keys()))]

# =========================
# PREDICT
# =========================
if st.button("🔍 Evaluate Risk"):

    input_data = {
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "SavingAccounts": saving,
        "CheckingAccount": checking,
        "CreditAmount": credit_amount,
        "Duration": duration,
        "Purpose": purpose
    }

    result = predict(model, input_data)

    prob = result["probability"]
    score = result["score"]

    st.subheader("📊 Risk Evaluation")

    st.metric("Credit Risk Score", f"{score}/100")
    st.progress(score / 100)

    if score > 70:
        st.error("🚨 HIGH RISK - Loan Rejected")
    elif score > 40:
        st.warning("⚠️ MEDIUM RISK - Manual Review")
    else:
        st.success("✅ LOW RISK - Loan Approved")

    st.write(f"**Probability of Default:** {prob:.2%}")

    # =========================
    # EXPLANATION
    # =========================
    st.subheader("🧠 Why this decision?")

    explanation = explain_prediction(model, input_data)

    for item in explanation:
        if item["impact"] > 0:
            st.write(f"🔴 {item['feature']} increases risk")
        else:
            st.write(f"🟢 {item['feature']} reduces risk")

    # =========================
    # MODEL COMPARISON
    # =========================
    st.subheader("📊 Model Comparison")

    st.write(f"Logistic Regression AUC: {metrics['log_auc']:.3f}")
    st.write(f"Random Forest AUC: {metrics['rf_auc']:.3f}")

    # =========================
    # ROC CURVE
    # =========================
    st.subheader("📈 ROC Curve")

    y_test = metrics["y_test"]
    X_test = metrics["X_test"]

    y_proba = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    st.pyplot(fig)

    st.caption("Risk evaluation powered by Machine Learning model")