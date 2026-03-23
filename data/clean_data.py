import pandas as pd

df = pd.read_csv("german_credit_data.csv")

# eliminar columna innecesaria
df = df.drop(columns=["Unnamed: 0"], errors="ignore")

# eliminar nulos
df = df.dropna()

# =========================
# CREAR TARGET (RISK)
# =========================
df["Risk"] = (
    (df["Credit amount"] > 5000) &
    (df["Checking account"].isin(["little", "moderate"])) &
    (df["Duration"] > 24)
).astype(int)

# =========================
# LIMPIEZA
# =========================
df = df.rename(columns={
    "Saving accounts": "SavingAccounts",
    "Checking account": "CheckingAccount",
    "Credit amount": "CreditAmount"
})

# convertir categóricos
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].astype("category").cat.codes

# guardar
df.to_csv("credit_clean.csv", index=False)

print("✅ Dataset limpio listo con Risk")