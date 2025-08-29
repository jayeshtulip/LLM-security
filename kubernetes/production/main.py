from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

app = FastAPI(title="Loan Prediction API", version="1.0.0")

class LoanApplication(BaseModel):
    Gender: str = Field(..., example="Male")
    Married: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="1")
    Education: str = Field(..., example="Graduate")
    Self_Employed: str = Field(..., example="No")
    ApplicantIncome: int = Field(..., example=5849)
    CoapplicantIncome: int = Field(..., example=0)
    LoanAmount: int = Field(..., example=120)
    Loan_Amount_Term: int = Field(..., example=360)
    Credit_History: int = Field(..., example=1)
    Property_Area: str = Field(..., example="Urban")

model = None

def init_model():
    global model
    np.random.seed(42)
    n = 1000
    data = {
        "ApplicantIncome": np.random.randint(1000, 15000, n),
        "Credit_History": np.random.choice([0, 1], n, p=[0.2, 0.8]),
        "LoanAmount": np.random.randint(50, 500, n)
    }
    df = pd.DataFrame(data)
    approval_prob = (df["Credit_History"] * 0.6 + (df["ApplicantIncome"] / 10000) * 0.4)
    df["Approved"] = (approval_prob > 0.5).astype(int)
    X = df[["ApplicantIncome", "Credit_History", "LoanAmount"]]
    y = df["Approved"]
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

@app.on_event("startup")
async def startup_event():
    init_model()

@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": datetime.utcnow(), "version": "1.0.0", "model_status": "loaded" if model is not None else "not_loaded"}

@app.get("/ready")
def ready():
    return {"status": "ready"}

@app.get("/")
def root():
    return {"message": "Loan Prediction API", "version": "1.0.0", "endpoints": ["/health", "/ready", "/predict", "/docs"]}

@app.post("/predict")
def predict(loan_app: LoanApplication):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        features = [[loan_app.ApplicantIncome, loan_app.Credit_History, loan_app.LoanAmount]]
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0]
        return {"prediction": int(prediction), "prediction_label": "Approved" if prediction == 1 else "Rejected", "probability": {"approved": float(probability[1]), "rejected": float(probability[0])}, "confidence": float(max(probability)), "model_version": "1.0.0"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predict/example")
def predict_example():
    sample = LoanApplication(Gender="Male", Married="Yes", Dependents="1", Education="Graduate", Self_Employed="No", ApplicantIncome=5849, CoapplicantIncome=0, LoanAmount=120, Loan_Amount_Term=360, Credit_History=1, Property_Area="Urban")
    return predict(sample)
