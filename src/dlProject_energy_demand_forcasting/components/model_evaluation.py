import os
import sys
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.dlProject_energy_demand_forcasting.utils.exception import CustomException
from src.dlProject_energy_demand_forcasting.utils.logger import logging
from src.dlProject_energy_demand_forcasting.utils.utils import load_object
from src.dlProject_energy_demand_forcasting.entity.config_entity import ModelEvaluationConfig

# ── Centralized MLflow tracking config ──────────────────────────────────
MLFLOW_TRACKING_URI = "https://dagshub.com/MainakDanOpRes/dlops_energy_forecasting_with_MLflow.mlflow"
MLFLOW_EXPERIMENT_NAME = "energy_demand_forecasting"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def create_sequences(self, data, window_size):
        """Same windowing logic as model_trainer, applied to the test set."""
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    def inverse_transform_target(self, values, preprocessor, target_col):
        """
        Inverts MinMaxScaler scaling for the target column only.
        Rebuilds a same-width dummy array, inverse-transforms, then slices
        back the target column.
        """
        scaler_step = preprocessor.named_steps["scaler"]
        min_max_scaler = scaler_step.scaler
        columns = list(scaler_step.columns)
        col_idx = columns.index(target_col)

        dummy = np.zeros((len(values), len(columns)))
        dummy[:, col_idx] = values.flatten()
        inv = min_max_scaler.inverse_transform(dummy)
        return inv[:, col_idx]

    def eval_metrics(self, actual, pred):
        rmse = float(np.sqrt(mean_squared_error(actual, pred)))
        mae  = float(mean_absolute_error(actual, pred))
        r2   = float(r2_score(actual, pred))
        mape = float(np.mean(np.abs((actual - pred) / np.clip(np.abs(actual), 1e-8, None))) * 100)
        return rmse, mae, r2, mape

    def initiate_model_evaluation(self):
        try:
            # ── load best model name from trainer artifact ─────────────────────
            best_model_info = load_object(
                file_path=os.path.join("artifacts", "model_trainer", "best_model_info.pkl")
            )
            best_model_name = best_model_info.get("model_name", "Unknown")
            logging.info(f"Best model used for evaluation: {best_model_name}")

            # ── load data ──────────────────────────────────────────────────────
            logging.info("Loading transformed test data for evaluation.")
            test_df = pd.read_csv(self.config.test_data_path, index_col="Datetime")

            target = self.config.target_column
            logging.info(f"Target column: {target}")
            test_array = test_df[target].values.reshape(-1, 1)

            logging.info(f"Creating sequences with window size: {self.config.window_size}")
            X_test, y_test_scaled = self.create_sequences(test_array, self.config.window_size)

            # ── load model & preprocessor ──────────────────────────────────────
            logging.info(f"Loading trained model from {self.config.model_path}")
            model = load_model(self.config.model_path, compile=False)

            logging.info(f"Loading preprocessor from {self.config.preprocessor_path}")
            preprocessor = load_object(file_path=self.config.preprocessor_path)

            # ── predict & inverse transform ────────────────────────────────────
            preds_scaled = model.predict(X_test)

            y_test_actual = self.inverse_transform_target(y_test_scaled, preprocessor, target)
            preds_actual  = self.inverse_transform_target(preds_scaled,  preprocessor, target)

            # ── metrics ────────────────────────────────────────────────────────
            rmse,   mae,   r2,   mape   = self.eval_metrics(y_test_actual,          preds_actual)
            rmse_s, mae_s, r2_s, mape_s = self.eval_metrics(y_test_scaled.flatten(), preds_scaled.flatten())

            # numeric-only dict for MLflow (strings go in params/tags)
            numeric_metrics = {
                "rmse": rmse, "mae": mae, "r2_score": r2, "mape": mape,
                "rmse_scaled": rmse_s, "mae_scaled": mae_s,
                "r2_score_scaled": r2_s, "mape_scaled": mape_s,
            }

            # full dict (with model name) for the JSON report
            full_metrics = {"model_name": best_model_name, **numeric_metrics}

            logging.info(f"Evaluation metrics (real units): {full_metrics}")

            # ── save metrics JSON ──────────────────────────────────────────────
            os.makedirs(self.config.root_dir, exist_ok=True)
            with open(self.config.metric_file_name, "w") as f:
                json.dump(full_metrics, f, indent=4)

            # ── MLflow logging ─────────────────────────────────────────────────
            mlflow.set_registry_uri(MLFLOW_TRACKING_URI)
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            logging.info(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")
            logging.info(f"MLflow run_name: {repr(best_model_name)}")

            with mlflow.start_run(run_name=best_model_name):
                # model name as tag and param (both visible in MLflow UI)
                mlflow.set_tag("model_name", best_model_name)

                base_params = {
                    "best_model":    best_model_name,
                    "window_size":   self.config.window_size,
                    "target_column": target,
                }

                # ── pull best-model hyperparameters, if stored in best_model_info ──
                excluded_keys = {"model_name", "model", "history"}  # non-scalar / duplicate keys
                best_model_params = {
                    k: v for k, v in best_model_info.items()
                    if k not in excluded_keys and isinstance(v, (str, int, float, bool))
                }

                if best_model_params:
                    logging.info(f"Best model hyperparameters: {best_model_params}")
                else:
                    logging.info("No hyperparameters found in best_model_info; falling back to model config.")
                    # Fallback: pull key architecture params straight from the Keras model
                    try:
                        best_model_params = {
                            "num_layers": len(model.layers),
                            "total_params": model.count_params(),
                            "optimizer": model.optimizer.__class__.__name__ if model.optimizer else "unknown",
                        }
                    except Exception as extract_err:
                        logging.warning(f"Could not extract fallback model params: {extract_err}")
                        best_model_params = {}

                all_params = {**base_params, **best_model_params}
                mlflow.log_params(all_params)

                mlflow.log_metrics(numeric_metrics)   # ← numeric only, no strings

                if tracking_url_type_store != "file":
                    mlflow.keras.log_model(
                        model, "model",
                        registered_model_name=f"EnergyDemandForecast_{best_model_name}"
                    )
                else:
                    mlflow.keras.log_model(
                        model, "model",
                        registered_model_name=f"EnergyDemandForecast_{best_model_name}"
                    )
                logging.info("Model evaluation completed and logged to MLflow.")
                return full_metrics

        except Exception as e:
            raise CustomException(e, sys)