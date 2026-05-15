import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

from src.model import train_model, predict, explain_prediction


st.set_page_config(
    page_title="Credit Risk AI",
    layout="centered",
)


@st.cache_resource
def load_model():
    return train_model()


model, metrics = load_model()


st.title("AI Credit Risk Scoring")

st.markdown(
    """
This application simulates a real-world credit risk system used in fintech and banking.  
It predicts the probability of default and supports loan approval decisions.
"""
)

st.subheader("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 75, 30)

    job_map = {
        "Unskilled": 0,
        "Skilled": 1,
        "Highly Skilled": 2,
        "Executive": 3,
    }
    job_label = st.selectbox("Job Level", list(job_map.keys()))
    job = job_map[job_label]

    housing_map = {
        "Rent": 0,
        "Own": 1,
        "Free": 2,
    }
    housing_label = st.selectbox("Housing", list(housing_map.keys()))
    housing = housing_map[housing_label]

with col2:
    credit_amount = st.number_input(
        "Credit Amount",
        min_value=100,
        max_value=20000,
        value=5000,
        step=100,
    )

    duration = st.slider("Duration (months)", 6, 72, 24)

    purpose_map = {
        "Car": 0,
        "Furniture": 1,
        "Education": 2,
        "Business": 3,
        "Appliances": 4,
        "Repairs": 5,
        "Vacation": 6,
        "Other": 7,
    }
    purpose_label = st.selectbox("Purpose", list(purpose_map.keys()))
    purpose = purpose_map[purpose_label]

saving_map = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Very High": 3,
}
saving_label = st.selectbox("Saving Accounts", list(saving_map.keys()))
saving = saving_map[saving_label]

checking_map = {
    "Low": 0,
    "Moderate": 1,
    "High": 2,
    "Very High": 3,
}
checking_label = st.selectbox("Checking Account", list(checking_map.keys()))
checking = checking_map[checking_label]

sex_map = {
    "Male": 1,
    "Female": 0,
}
sex_label = st.selectbox("Sex", list(sex_map.keys()))
sex = sex_map[sex_label]


if st.button("Evaluate Risk"):
    input_data = {
        "Age": age,
        "Sex": sex,
        "Job": job,
        "Housing": housing,
        "SavingAccounts": saving,
        "CheckingAccount": checking,
        "CreditAmount": credit_amount,
        "Duration": duration,
        "Purpose": purpose,
    }

    result = predict(model, input_data)

    probability = result["probability"]
    score = result["score"]
    decision = result["decision"]

    st.subheader("Risk Evaluation")

    st.metric("Credit Risk Score", f"{score}/850")
    st.progress((score - 300) / 550)

    if decision == "REJECT":
        st.error("High Risk - Loan Rejected")
    elif decision == "REVIEW":
        st.warning("Medium Risk - Manual Review")
    else:
        st.success("Low Risk - Loan Approved")

    st.write(f"**Probability of Default:** {probability:.2%}")

    st.subheader("Why this decision?")

    explanation = explain_prediction(model, input_data)

    if explanation:
        for item in explanation:
            if item["impact"] > 0:
                st.write(f"{item['feature']} increases risk")
            else:
                st.write(f"{item['feature']} reduces risk")
    else:
        st.write("Feature-level explanation is not available for this model.")

    st.subheader("Model Comparison")

    st.write(f"Logistic Regression AUC: {metrics['log_auc']:.3f}")
    st.write(f"Random Forest AUC: {metrics['rf_auc']:.3f}")

    st.subheader("ROC Curve")

    y_test = metrics["y_test"]
    x_test = metrics["X_test"]

    y_proba = model.predict_proba(x_test)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="Model")
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend()

    st.pyplot(fig)

    st.caption("Risk evaluation powered by machine learning.")