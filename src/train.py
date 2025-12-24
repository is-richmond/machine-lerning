#!/usr/bin/env python3
"""
train.py

Тренировка моделей на датасете California Housing (sklearn).
Сохраняет scaler и модели в models/, метрики в outputs/metrics.json и пример test-rows.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except Exception:
    LGB_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

def prepare_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    feature_names = data.feature_names
    target_name = data.target.name
    return df, feature_names, target_name

def compute_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2)}

def train_and_evaluate(random_state=42):
    df, feature_names, target_name = load_data()
    X = df[feature_names]
    y = df[target_name]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # Baseline: Linear Regression
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    results["LinearRegression"] = compute_metrics(y_test, y_pred_lr)

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    results["RandomForest"] = compute_metrics(y_test, y_pred_rf)

    # LightGBM (если доступен)
    if LGB_AVAILABLE:
        lgbm = lgb.LGBMRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
        lgbm.fit(X_train_scaled, y_train)
        y_pred_lgbm = lgbm.predict(X_test_scaled)
        results["LightGBM"] = compute_metrics(y_test, y_pred_lgbm)
    else:
        results["LightGBM"] = {"info": "lightgbm not installed or failed to import"}

    # Сохраняем scaler и модели
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(lr, MODELS_DIR / "linear_model.joblib")
    joblib.dump(rf, MODELS_DIR / "rf_model.joblib")
    if LGB_AVAILABLE:
        joblib.dump(lgbm, MODELS_DIR / "lgbm_model.joblib")

    # Сохраняем метрики и список признаков
    metrics = {
        "models": results,
        "features": list(feature_names)
    }
    with open(OUTPUTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Сохраним sample test rows
    sample = X_test.copy()
    sample["target"] = y_test
    sample.sample(20, random_state=42).to_csv(OUTPUTS_DIR / "sample_test_rows.csv", index=False)

    print("Training finished. Metrics:")
    print(json.dumps(metrics, indent=2))

    return metrics

if __name__ == "__main__":
    prepare_dirs()
    train_and_evaluate()