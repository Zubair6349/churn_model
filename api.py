
import joblib
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel

model = joblib.load("churn_model.pkl")
threshold = joblib.load("churn_threshold.pkl")

app = FastAPI(title="Customer Churn Prediction API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Customer(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict_churn(customer: Customer):

    customer_data = pd.DataFrame([customer.model_dump()])

    probability = model.predict_proba(customer_data)[0][1]

    prediction = int(probability >= threshold)

    return {
        "churn_probability": round(float(probability), 4),
        "prediction": prediction,
        "risk": "High" if prediction == 1 else "Low",
        "threshold": threshold
    }


@app.get("/")
def home():
    return {"message": "Customer Churn Prediction API is running!"}
