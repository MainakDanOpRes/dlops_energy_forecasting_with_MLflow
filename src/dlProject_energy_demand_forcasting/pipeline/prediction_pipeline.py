import os
import sys
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

from src.dlProject_energy_demand_forcasting.utils.exception import CustomException
from src.dlProject_energy_demand_forcasting.utils.logger import logging
from src.dlProject_energy_demand_forcasting.utils.utils import load_object


# ── Paths (relative to project root) ──────────────────────────────────────────
MODEL_PATH       = os.path.join("artifacts", "model_trainer", "model.h5")
PREPROCESSOR_PATH = os.path.join("artifacts", "data_transformation", "preprocessor.pkl")
BEST_MODEL_PATH  = os.path.join("artifacts", "model_trainer", "best_model_info.pkl")
WINDOW_SIZE      = 24          
TARGET_COLUMN    = "Global_active_power"


class PredictionPipeline:
    """
    Loads the trained model + preprocessor once, then serves predictions.

    Two usage modes:
      1. predict_from_raw(df)   — accepts a raw DataFrame with the original
                                   columns (Date, Time, plus features).
                                   Runs the FULL preprocessor pipeline on it.

      2. predict_from_scaled(series) — accepts a pandas Series / 1-D array of
                                        already-scaled values (e.g. loaded directly
                                        from test_transformed.csv).
                                        Skips preprocessing; just windows & predicts.
    """

    def __init__(self):
        try:
            logging.info("Loading model and preprocessor for prediction.")
            self.model        = load_model(MODEL_PATH, compile=False)
            self.preprocessor = load_object(file_path=PREPROCESSOR_PATH)
            self.best_info    = load_object(file_path=BEST_MODEL_PATH)
            self.model_name   = self.best_info.get("model_name", "Unknown")
            logging.info(f"Loaded model: {self.model_name}")
        except Exception as e:
            raise CustomException(e, sys)

    # ── helpers ────────────────────────────────────────────────────────────────

    def _create_sequences(self, array_1d: np.ndarray) -> np.ndarray:
        """Build sliding-window sequences [samples, window_size, 1]."""
        X = []
        for i in range(len(array_1d) - WINDOW_SIZE):
            X.append(array_1d[i : i + WINDOW_SIZE])
        return np.array(X).reshape(-1, WINDOW_SIZE, 1)

    def _inverse_transform(self, scaled_values: np.ndarray) -> np.ndarray:
        """
        Invert MinMaxScaler for the target column only.
        The scaler was fit on ALL numeric columns, so we build a dummy
        same-width array, fill the target column, inverse-transform, then slice.
        """
        scaler_step = self.preprocessor.named_steps["scaler"]
        mms         = scaler_step.scaler
        columns     = list(scaler_step.columns)
        col_idx     = columns.index(TARGET_COLUMN)

        dummy = np.zeros((len(scaled_values), len(columns)))
        dummy[:, col_idx] = scaled_values.flatten()
        inv = mms.inverse_transform(dummy)
        return inv[:, col_idx]

    # ── public API ─────────────────────────────────────────────────────────────

    def predict_from_raw(self, df: pd.DataFrame) -> dict:
        """
        Full pipeline: raw DataFrame → preprocessor → sequences → model → inverse.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at minimum Date, Time, and all numeric feature columns
            as they appear in the original household_power_consumption.txt.

        Returns
        -------
        dict with keys: model_name, predictions (real units), timestamps
        """
        try:
            logging.info("Running full preprocessing on raw input.")
            transformed = self.preprocessor.transform(df)

            scaled_array = transformed[TARGET_COLUMN].values
            X            = self._create_sequences(scaled_array)

            preds_scaled = self.model.predict(X).flatten()
            preds_actual = self._inverse_transform(preds_scaled)

            timestamps = transformed.index[WINDOW_SIZE:].tolist()

            return {
                "model_name"  : self.model_name,
                "timestamps"  : [str(t) for t in timestamps],
                "predictions" : preds_actual.tolist(),
            }
        except Exception as e:
            raise CustomException(e, sys)

    def predict_from_scaled(self, scaled_series: pd.Series | np.ndarray) -> dict:
        """
        Lightweight path: already-scaled 1-D values → sequences → model → inverse.

        Parameters
        ----------
        scaled_series : array-like of shape (N,)
            Already-scaled Global_active_power values from test_transformed.csv.
            Must contain at least (WINDOW_SIZE + 1) values.

        Returns
        -------
        dict with keys: model_name, predictions (real units), predictions_scaled
        """
        try:
            logging.info("Running prediction on pre-scaled input.")
            array = np.array(scaled_series).flatten()

            if len(array) < WINDOW_SIZE + 1:
                raise ValueError(
                    f"Need at least {WINDOW_SIZE + 1} scaled values; got {len(array)}."
                )

            X            = self._create_sequences(array)
            preds_scaled = self.model.predict(X).flatten()
            preds_actual = self._inverse_transform(preds_scaled)

            return {
                "model_name"        : self.model_name,
                "predictions"       : preds_actual.tolist(),
                "predictions_scaled": preds_scaled.tolist(),
            }
        except Exception as e:
            raise CustomException(e, sys)

    def predict_next_n(self, last_window: list[float], n_steps: int = 24) -> dict:
        """
        Autoregressive multi-step forecast.
        Given the last WINDOW_SIZE scaled values, predict the next n_steps hours.

        Parameters
        ----------
        last_window : list of float (length == WINDOW_SIZE)
            The most recent WINDOW_SIZE scaled target values.
        n_steps : int
            How many future steps to forecast.

        Returns
        -------
        dict with keys: model_name, steps, predictions (real units), predictions_scaled
        """
        try:
            if len(last_window) != WINDOW_SIZE:
                raise ValueError(
                    f"last_window must have exactly {WINDOW_SIZE} values; got {len(last_window)}."
                )

            window       = list(last_window)
            preds_scaled = []

            for _ in range(n_steps):
                x          = np.array(window[-WINDOW_SIZE:]).reshape(1, WINDOW_SIZE, 1)
                next_scaled = float(self.model.predict(x, verbose=0)[0][0])
                preds_scaled.append(next_scaled)
                window.append(next_scaled)

            preds_actual = self._inverse_transform(np.array(preds_scaled))

            return {
                "model_name"        : self.model_name,
                "steps"             : n_steps,
                "predictions"       : preds_actual.tolist(),
                "predictions_scaled": preds_scaled,
            }
        except Exception as e:
            raise CustomException(e, sys)