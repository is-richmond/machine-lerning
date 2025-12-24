#!/usr/bin/env python3
"""
FastAPI приложение: предоставляет endpoint /predict и /metrics
Запуск: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Literal, Optional
from src.predict import predict_from_dict, load_metrics

app = FastAPI(
    title="House Price Prediction API",
    description="API для прогнозирования медианной стоимости жилья (California Housing).",
    version="1.0.0"
)

class PredictRequest(BaseModel):
    model: Optional[Literal["rf", "lr", "lgbm"]] = "rf"
    features: Dict[str, float]

class PredictResponse(BaseModel):
    model: str
    prediction: float
    unit_notes: Optional[str] = "target is MedianHouseValue in dataset units"
    metrics: Optional[dict] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metrics")
def metrics():
    return load_metrics()

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Валидация простая: проверяем, что features содержит нужные ключи (из metrics)
    metrics = load_metrics()
    feature_names = metrics.get("features")
    if feature_names:
        missing = [f for f in feature_names if f not in req.features]
        if missing:
            raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
    try:
        pred = predict_from_dict(req.features, model_name=req.model)
        return PredictResponse(model=req.model, prediction=pred, metrics=metrics.get("models"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))