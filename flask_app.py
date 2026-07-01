"""
Flask serving layer for the Energy Demand Forecasting model.

Run:
    python flask_app.py

Routes
------
GET  /              → health check (JSON)
GET  /model-info    → best model name + r2 from training
POST /predict       → predict from scaled values list
POST /forecast      → autoregressive n-step future forecast
POST /predict-raw   → predict from raw CSV-style rows
"""

import sys
import traceback

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

from src.dlProject_energy_demand_forcasting.utils.logger import logging
from src.dlProject_energy_demand_forcasting.utils.exception import CustomException
from src.dlProject_energy_demand_forcasting.pipeline.prediction_pipeline import (
    PredictionPipeline,
    WINDOW_SIZE,
    TARGET_COLUMN,
)

app  = Flask(__name__)
CORS(app)

# ── Load pipeline once ─────────────────────────────────────────────────────────
logging.info("Loading PredictionPipeline for Flask app.")
pipeline = PredictionPipeline()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": pipeline.model_name})


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify({
        "model_name" : pipeline.model_name,
        "r2_score"   : pipeline.best_info.get("r2_score"),
        "window_size": WINDOW_SIZE,
        "target"     : TARGET_COLUMN,
    })


@app.route("/predict", methods=["POST"])
def predict_scaled():
    """
    Body (JSON):
        { "scaled_values": [0.3, 0.31, ...] }   # >= WINDOW_SIZE+1 values

    Returns real-unit (kW) predictions.
    """
    try:
        body          = request.get_json(force=True)
        scaled_values = body.get("scaled_values", [])

        if len(scaled_values) < WINDOW_SIZE + 1:
            return jsonify({
                "error": f"Need at least {WINDOW_SIZE + 1} scaled values; got {len(scaled_values)}."
            }), 400

        result = pipeline.predict_from_scaled(np.array(scaled_values))
        return jsonify(result)

    except CustomException as e:
        return jsonify({"error": str(e)}), 500
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/forecast", methods=["POST"])
def forecast():
    """
    Body (JSON):
        {
          "last_window": [0.3, 0.31, ...],   # exactly WINDOW_SIZE values
          "n_steps": 24                       # optional, default 24
        }

    Returns n_steps future real-unit predictions.
    """
    try:
        body        = request.get_json(force=True)
        last_window = body.get("last_window", [])
        n_steps     = int(body.get("n_steps", 24))

        if len(last_window) != WINDOW_SIZE:
            return jsonify({
                "error": f"last_window must have exactly {WINDOW_SIZE} values; got {len(last_window)}."
            }), 400

        if not (1 <= n_steps <= 168):
            return jsonify({"error": "n_steps must be between 1 and 168."}), 400

        result = pipeline.predict_next_n(last_window=last_window, n_steps=n_steps)
        return jsonify(result)

    except CustomException as e:
        return jsonify({"error": str(e)}), 500
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/predict-raw", methods=["POST"])
def predict_raw():
    """
    Body (JSON):
        { "rows": [ { "Date": "16/12/2006", "Time": "17:24:00",
                      "Global_active_power": 4.216, ... }, ... ] }

    Runs the full preprocessor pipeline then returns real-unit predictions.
    """
    try:
        body = request.get_json(force=True)
        rows = body.get("rows", [])

        if len(rows) < WINDOW_SIZE + 1:
            return jsonify({
                "error": f"Need at least {WINDOW_SIZE + 1} raw rows; got {len(rows)}."
            }), 400

        df     = pd.DataFrame(rows)
        result = pipeline.predict_from_raw(df)
        return jsonify(result)

    except CustomException as e:
        return jsonify({"error": str(e)}), 500
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)