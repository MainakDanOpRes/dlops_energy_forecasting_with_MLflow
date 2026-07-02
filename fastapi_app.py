"""
FastAPI serving layer for the Energy Demand Forecasting model.

Run:
    uvicorn fastapi_app:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
GET  /              → health check
GET  /model-info    → best model name + r2 from training
POST /predict       → predict from last window of scaled values
POST /forecast      → autoregressive n-step future forecast
POST /predict-raw   → predict from raw (unscaled) CSV rows
POST /train         → kick off a retraining run (main.py) in the background
GET  /train/status  → is training running, last exit code
GET  /train/logs    → tail of the training subprocess output
"""

import os
import subprocess
import sys
import threading
import traceback
from typing import List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.dlProject_energy_demand_forcasting.utils.logger import logging
from src.dlProject_energy_demand_forcasting.utils.exception import CustomException
from src.dlProject_energy_demand_forcasting.pipeline.prediction_pipeline import (
    PredictionPipeline,
    WINDOW_SIZE,
    TARGET_COLUMN,
)

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Energy Demand Forecasting API",
    description="LSTM/GRU forecasting on UCI Household Power Consumption dataset.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves static/index.html at /dashboard/ (and any other files placed in static/)
app.mount("/dashboard", StaticFiles(directory="static", html=True), name="dashboard")

# ── Load pipeline once at startup ──────────────────────────────────────────────
pipeline: PredictionPipeline | None = None

@app.on_event("startup")
def load_pipeline():
    global pipeline
    logging.info("Initialising PredictionPipeline at startup.")
    pipeline = PredictionPipeline()


# ── Background training state ───────────────────────────────────────────────────
# Training runs main.py as a subprocess in a background thread so it doesn't block
# the event loop. Only one run is allowed at a time. Logs are kept in memory (last
# 500 lines) so the frontend can poll them while training is in progress.
_training_lock = threading.Lock()
_training_state = {
    "running": False,
    "returncode": None,
    "logs": [],
}


def _run_training():
    global pipeline
    my_env = os.environ.copy()
    my_env["PYTHONIOENCODING"] = "utf-8"

    with _training_lock:
        _training_state["running"] = True
        _training_state["returncode"] = None
        _training_state["logs"] = []

    try:
        proc = subprocess.Popen(
            [sys.executable, "main.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="replace",
            env=my_env,
            bufsize=1,
        )
        for line in proc.stdout:
            with _training_lock:
                _training_state["logs"].append(line.rstrip("\n"))
                _training_state["logs"] = _training_state["logs"][-500:]
        proc.wait()
        returncode = proc.returncode
    except Exception:
        returncode = -1
        with _training_lock:
            _training_state["logs"].append(traceback.format_exc())

    with _training_lock:
        _training_state["running"] = False
        _training_state["returncode"] = returncode

    # Reload the model into memory so /predict and /forecast pick up the new
    # weights without needing a container restart.
    if returncode == 0:
        try:
            pipeline = PredictionPipeline()
            with _training_lock:
                _training_state["logs"].append("[api] Pipeline reloaded with newly trained model.")
        except Exception:
            with _training_lock:
                _training_state["logs"].append("[api] Training succeeded but reloading the pipeline failed:")
                _training_state["logs"].append(traceback.format_exc())


# ── Pydantic schemas ───────────────────────────────────────────────────────────

class ScaledWindowRequest(BaseModel):
    """Provide >= WINDOW_SIZE already-scaled Global_active_power values."""
    scaled_values: List[float] = Field(
        ...,
        min_length=WINDOW_SIZE + 1,
        description=f"At least {WINDOW_SIZE + 1} scaled values from test_transformed.csv",
        example=[0.3] * (WINDOW_SIZE + 5),
    )

class ForecastRequest(BaseModel):
    """Provide exactly WINDOW_SIZE scaled values; get n_steps future predictions."""
    last_window: List[float] = Field(
        ...,
        min_length=WINDOW_SIZE,
        max_length=WINDOW_SIZE,
        description=f"Exactly {WINDOW_SIZE} most-recent scaled values.",
        example=[0.3] * WINDOW_SIZE,
    )
    n_steps: int = Field(default=24, ge=1, le=168, description="Hours to forecast (max 168 = 1 week).")

class RawRow(BaseModel):
    """One raw data row matching the original dataset columns."""
    Date: str                       = Field(..., example="16/12/2006")
    Time: str                       = Field(..., example="17:24:00")
    Global_active_power: float      = Field(..., example=4.216)
    Global_reactive_power: float    = Field(..., example=0.418)
    Voltage: float                  = Field(..., example=234.84)
    Global_intensity: float         = Field(..., example=18.4)
    Sub_metering_1: float           = Field(..., example=0.0)
    Sub_metering_2: float           = Field(..., example=1.0)
    Sub_metering_3: float           = Field(..., example=17.0)

class RawPredictRequest(BaseModel):
    rows: List[RawRow] = Field(..., min_length=WINDOW_SIZE + 1)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
def health():
    return {"status": "ok", "model": pipeline.model_name if pipeline else "not loaded"}


@app.get("/model-info", tags=["Info"])
def model_info():
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not loaded.")
    return {
        "model_name"  : pipeline.model_name,
        "r2_score"    : pipeline.best_info.get("r2_score"),
        "window_size" : WINDOW_SIZE,
        "target"      : TARGET_COLUMN,
    }


@app.post("/predict", tags=["Predict"])
def predict_scaled(request: ScaledWindowRequest):
    """
    Predict from already-scaled values (from test_transformed.csv).
    Returns real-unit predictions (kW).
    """
    try:
        result = pipeline.predict_from_scaled(np.array(request.scaled_values))
        return result
    except CustomException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/forecast", tags=["Predict"])
def forecast(request: ForecastRequest):
    """
    Autoregressive multi-step forecast.
    Feed the last {WINDOW_SIZE} scaled values; get n_steps future hours.
    """
    try:
        result = pipeline.predict_next_n(
            last_window=request.last_window,
            n_steps=request.n_steps,
        )
        return result
    except CustomException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/predict-raw", tags=["Predict"])
def predict_raw(request: RawPredictRequest):
    """
    Predict from raw (unscaled) input rows.
    The full preprocessor pipeline (DatetimeIndexer → Imputer → Resampler → Scaler)
    is applied before inference.
    """
    try:
        df = pd.DataFrame([row.model_dump() for row in request.rows])
        result = pipeline.predict_from_raw(df)
        return result
    except CustomException as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=traceback.format_exc())


@app.post("/train", tags=["Train"])
def start_training():
    """
    Kick off `main.py` (the training pipeline) as a background subprocess.
    Returns immediately; poll /train/status and /train/logs for progress.
    """
    with _training_lock:
        if _training_state["running"]:
            raise HTTPException(status_code=409, detail="Training is already in progress.")

    thread = threading.Thread(target=_run_training, daemon=True)
    thread.start()
    return {"status": "started"}


@app.get("/train/status", tags=["Train"])
def train_status():
    with _training_lock:
        return {
            "running": _training_state["running"],
            "returncode": _training_state["returncode"],
        }


@app.get("/train/logs", tags=["Train"])
def train_logs():
    with _training_lock:
        return {"logs": _training_state["logs"]}