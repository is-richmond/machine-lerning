#!/usr/bin/env python3
"""
FastAPI приложение: предоставляет endpoint /predict и /metrics
Запуск: uvicorn src.api:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import Literal, Optional
from src.predict import predict_from_dict, load_metrics

app = FastAPI(
    title="House Price Prediction API",
    description="API для прогнозирования медианной стоимости жилья (California Housing).",
    version="1.0.0",
)


class PredictResponse(BaseModel):
    model: str
    prediction: float
    prediction_usd: float
    unit: str = "Hundreds of thousands USD"
    description: str = "Predicted median house value. Example: 3.15 = $315,000"
    metrics: Optional[dict] = None


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return load_metrics()


@app.get("/predict", response_model=PredictResponse)
def predict(
    model: Literal["rf", "lr", "lgbm"] = Query("rf"),
    MedInc: float = Query(..., description="Median Income"),
    HouseAge: float = Query(..., description="House Age"),
    AveRooms: float = Query(..., description="Average Rooms"),
    AveBedrms: float = Query(..., description="Average Bedrooms"),
    Population: float = Query(..., description="Population"),
    AveOccup: float = Query(..., description="Average Occupancy"),
    Latitude: float = Query(..., description="Latitude"),
    Longitude: float = Query(..., description="Longitude"),
):
    # Build features dict from query parameters
    features = {
        "MedInc": MedInc,
        "HouseAge": HouseAge,
        "AveRooms": AveRooms,
        "AveBedrms": AveBedrms,
        "Population": Population,
        "AveOccup": AveOccup,
        "Latitude": Latitude,
        "Longitude": Longitude,
    }

    try:
        pred = predict_from_dict(features, model_name=model)
        metrics = load_metrics()
        return PredictResponse(
            model=model,
            prediction=pred,
            prediction_usd=round(pred * 100_000, 2),
            metrics=metrics.get("models"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
