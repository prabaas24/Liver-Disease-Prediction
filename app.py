from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib
import numpy as np

app=FastAPI(title="Liver Disease Prediction API")

model = joblib.load("Liver_Prediction_SVM.pkl")
label_map = {
    0: "cirrhosis",
    1: "fibrosis",
    2: "hepatitis",
    3: "no_disease",
    4: "suspect_disease"
}

class PredictionInput(BaseModel):
    features: List[float]

@app.get("/")
def home():
    return{"status":"API is running"}

@app.post("/predict")
def predict(input:PredictionInput):
    data=np.array(input.features).reshape(1,-1)
    prediction=model.predict(data)
    pred_code = int(prediction[0])

    return{"prediction_code": pred_code,
    "prediction_label": label_map[pred_code]}
