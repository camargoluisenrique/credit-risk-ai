# Credit Risk Decision System

A credit risk system designed to estimate probability of default and support loan approval decisions.

This project focuses on translating model predictions into clear and consistent decisions, rather than treating prediction as the final output.

---

## Demo

The application allows real-time evaluation of applicants, including probability of default, credit score, and decision outcome.

## Live Demo

https://credit-risk-ai-livduk4ttdjaeyvx8vl5h6.streamlit.app/

---

## Problem Context

Credit risk assessment is a core part of financial systems. The goal is not only to estimate the likelihood of default, but to turn that estimate into decisions that are consistent and explainable.

In practice, a model is expected to:

- Estimate probability of default  
- Provide a usable risk score  
- Support decision rules (approve, review, reject)  
- Remain interpretable for audit and control  

---

## Approach

The project follows a straightforward machine learning workflow:

- Data preprocessing and feature preparation  
- Training multiple models (Logistic Regression and Random Forest)  
- Model selection based on ROC AUC  
- Probability-based predictions  
- Mapping predictions to scores and decisions  

On top of the model output, a decision layer is applied to simulate how credit policies are typically enforced.

---

## Model

Two models are trained and compared:

- Logistic Regression (baseline, easier to interpret)  
- Random Forest (captures non-linear relationships)  

The final model is selected based on validation performance.

---

## Risk Scoring

Predicted probabilities are converted into a credit score:

- Scale: 300 – 850  
- Higher score indicates lower risk  

This makes the output easier to interpret and closer to real credit scoring systems.

---

## Decision Logic

Instead of using raw predictions, the system applies a simple decision layer:

- Approve: low probability of default  
- Review: intermediate risk  
- Reject: high probability of default  

Separating prediction from decision is important in real systems, where business rules are applied on top of model outputs.

---

## Model Evaluation

Model performance is evaluated using:

- ROC AUC  
- Confusion Matrix  
- Distribution of predicted probabilities  

The selected model shows good separation between classes while remaining stable.

---

## Explainability

The system provides a basic explanation of each prediction by ranking features according to their impact.

This is not meant to be a full interpretability solution, but it gives a quick indication of what is driving the result.

---

## Application

The application provides a simple interface where a user can:

- Enter applicant information  
- Obtain probability of default  
- View a credit score  
- Receive a decision recommendation  
- See which variables influenced the result  

This represents a simplified version of a credit underwriting workflow.

---

## Project Structure

credit-risk-ai/
│
├── app.py
├── README.md
├── requirements.txt
│
├── src/
│     └── model.py
│
├── data/
│     └── credit_clean.csv
│
├── outputs/
│     └── models/
│
└── notebooks/



---

## Key Considerations

- Clear separation between prediction and decision  
- Use of an interpretable baseline model  
- Avoiding unnecessary complexity  
- Focus on decision-making rather than raw accuracy  

---

## Potential Extensions

- API integration (FastAPI)  
- Cost-sensitive optimization  
- Advanced explainability (SHAP)  
- Real-time scoring pipeline  

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

Author

Luis Enrique Camargo Rangel
Data Scientist | Applied Machine Learning


