#!/usr/bin/env python3
"""
predict.py

Утилиты: загрузка моделей и предсказание по входному словарю признаков.
"""

from pathlib import Path
import joblib
import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

def load_models():
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    lr = joblib.load(MODELS_DIR / "linear_model.joblib")
    rf = joblib.load(MODELS_DIR / "rf_model.joblib")
    lgbm = None
    try:
        lgbm = joblib.load(MODELS_DIR / "lgbm_model.joblib")
    except Exception:
        pass
    return scaler, lr, rf, lgbm

def load_metrics():
    try:
        with open(OUTPUTS_DIR / "metrics.json", "r") as f:
            return json.load(f)
    except Exception:
        return {}

def predict_from_dict(features: dict, model_name="rf"):
    """
    features: dict с ключами как в California Housing:
      ['MedInc','HouseAge','AveRooms','AveBedrms','Population','AveOccup','Latitude','Longitude']
    model_name: 'rf' | 'lr' | 'lgbm'
    """
    scaler, lr, rf, lgbm = load_models()
    df = pd.DataFrame([features])
    X_scaled = scaler.transform(df)
    if model_name == "lr":
        pred = lr.predict(X_scaled)[0]
    elif model_name == "lgbm" and lgbm is not None:
        pred = lgbm.predict(X_scaled)[0]
    else:
        pred = rf.predict(X_scaled)[0]
    return float(pred)