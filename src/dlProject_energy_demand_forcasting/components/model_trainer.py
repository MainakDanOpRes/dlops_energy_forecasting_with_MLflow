import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import keras_tuner as kt
import mlflow
import mlflow.tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from sklearn.metrics import r2_score

from src.dlProject_energy_demand_forcasting.utils.exception import CustomException
from src.dlProject_energy_demand_forcasting.utils.logger import logging
from src.dlProject_energy_demand_forcasting.utils.utils import save_object
from src.dlProject_energy_demand_forcasting.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def create_sequences(self, data, window_size):
        """
        Converts a 1D array into 3D sequences [samples, window_size, 1].
        """
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i : i + window_size])
            y.append(data[i + window_size])
        return np.array(X), np.array(y)

    def _make_hypermodel(self, model_type, input_shape):
        """
        Returns a build_fn(hp) closure that Keras Tuner calls once per trial.
        Search space is read from params.yaml (self.config.all_model_params[model_type]).
        """
        p = self.config.all_model_params[model_type]

        def build_fn(hp):
            units_1 = hp.Int(
                f"{model_type}_units_layer_1",
                min_value=p.units_layer_1.min,
                max_value=p.units_layer_1.max,
                step=p.units_layer_1.step,
            )
            units_2 = hp.Int(
                f"{model_type}_units_layer_2",
                min_value=p.units_layer_2.min,
                max_value=p.units_layer_2.max,
                step=p.units_layer_2.step,
            )
            dropout_rate = hp.Float(
                f"{model_type}_dropout_rate",
                min_value=p.dropout_rate.min,
                max_value=p.dropout_rate.max,
                step=p.dropout_rate.step,
            )
            dense_units = hp.Int(
                f"{model_type}_dense_units",
                min_value=p.dense_units.min,
                max_value=p.dense_units.max,
                step=p.dense_units.step,
            )
            learning_rate = hp.Choice(
                f"{model_type}_learning_rate", values=list(p.learning_rate["values"])
            )
            # batch_size is tuned but applied at .fit() time, not build time.
            # We stash it as a hyperparameter so it still shows up in trial logs.
            hp.Choice(f"{model_type}_batch_size", values=list(p.batch_size["values"]))

            model = Sequential([Input(shape=input_shape)])

            if model_type.upper() == "LSTM":
                model.add(LSTM(units=units_1, return_sequences=True))
                model.add(Dropout(dropout_rate))
                model.add(LSTM(units=units_2))
            elif model_type.upper() == "GRU":
                model.add(GRU(units=units_1, return_sequences=True))
                model.add(Dropout(dropout_rate))
                model.add(GRU(units=units_2))
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

            model.add(Dense(dense_units, activation='relu'))
            model.add(Dense(units=1))

            opt = tf.keras.optimizers.get(p.optimizer)
            opt.learning_rate = learning_rate
            model.compile(optimizer=opt, loss=p.loss, metrics=['mae'])
            return model

        return build_fn

    def _get_batch_size_for_trial(self, hp_values, model_type, default):
        """Keras Tuner stores every hp.* value in trial.hyperparameters.values."""
        return hp_values.get(f"{model_type}_batch_size", default)

    def _build_tuner(self, model_type, input_shape):
        t = self.config.tuning_params
        build_fn = self._make_hypermodel(model_type, input_shape)
        project_name = f"{model_type.lower()}_tuning"

        if t.tuner_type.lower() == "hyperband":
            tuner = kt.Hyperband(
                build_fn,
                objective=t.objective,
                max_epochs=t.hyperband_max_epochs,
                factor=3,
                seed=t.seed,
                directory=str(self.config.tuner_dir),
                project_name=project_name,
                overwrite=True,
            )
        else:
            tuner = kt.RandomSearch(
                build_fn,
                objective=t.objective,
                max_trials=t.max_trials,
                executions_per_trial=t.executions_per_trial,
                seed=t.seed,
                directory=str(self.config.tuner_dir),
                project_name=project_name,
                overwrite=True,
            )
        return tuner

    def _tune_model_type(self, model_type, X_train, y_train, X_test, y_test):
        """
        Runs the hyperparameter search for one model type (LSTM or GRU),
        logs every trial to MLflow as a nested run, retrains the best
        configuration fully, and returns (model, r2, best_hp_dict, val_loss, val_mae).
        """
        t = self.config.tuning_params
        input_shape = (self.config.window_size, 1)

        logging.info(f">>> Starting hyperparameter search for: {model_type}")
        tuner = self._build_tuner(model_type, input_shape)

        with mlflow.start_run(run_name=f"{model_type}_tuning", nested=True):
            mlflow.log_params({
                "model_type": model_type,
                "tuner_type": t.tuner_type,
                "objective": t.objective,
                "max_trials": t.get("max_trials", "n/a"),
                "tuner_epochs": t.tuner_epochs,
            })

            tuner.search(
                X_train, y_train,
                epochs=t.tuner_epochs,
                validation_data=(X_test, y_test),
                batch_size=32,  # default during search; per-trial batch_size handled below
                verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=t.patience)],
            )

            # Log every completed trial as its own nested run for full visibility in MLflow
            for trial_id, trial in tuner.oracle.trials.items():
                if trial.score is None:
                    continue
                with mlflow.start_run(run_name=f"{model_type}_trial_{trial_id}", nested=True):
                    mlflow.log_params(trial.hyperparameters.values)
                    mlflow.log_metric(t.objective, trial.score)

            best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
            best_batch_size = self._get_batch_size_for_trial(
                best_hp.values, model_type, default=self.config.all_model_params[model_type].batch_size["values"][0]
            )

            logging.info(f"Best {model_type} hyperparameters: {best_hp.values}")

            # Retrain the best configuration fully (tuner_epochs was a short proxy search budget)
            best_model = tuner.hypermodel.build(best_hp)
            history = best_model.fit(
                X_train, y_train,
                epochs=t.final_epochs,
                batch_size=best_batch_size,
                validation_data=(X_test, y_test),
                verbose=0,
                callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=t.patience, restore_best_weights=True)],
            )

            preds = best_model.predict(X_test, verbose=0)
            r2 = r2_score(y_test, preds)
            val_loss = min(history.history['val_loss'])
            val_mae = min(history.history['val_mae'])

            mlflow.log_params(best_hp.values)
            mlflow.log_param(f"{model_type}_final_epochs_ran", len(history.history['loss']))
            mlflow.log_metrics({
                "val_loss": val_loss,
                "val_mae": val_mae,
                "r2_score": r2,
            })

            logging.info(f"{model_type} best config -> R2: {r2:.4f}, val_loss: {val_loss:.4f}")

        return best_model, r2, dict(best_hp.values), val_loss, val_mae

    def initiate_model_trainer(self):
        try:
            logging.info("Loading transformed train and test data.")
            train_df = pd.read_csv(self.config.train_data_path, index_col='Datetime')
            test_df = pd.read_csv(self.config.test_data_path, index_col='Datetime')

            target = self.config.target_column
            train_array = train_df[target].values.reshape(-1, 1)
            test_array = test_df[target].values.reshape(-1, 1)

            logging.info(f"Creating 3D sequences with window size: {self.config.window_size}")
            X_train, y_train = self.create_sequences(train_array, self.config.window_size)
            X_test, y_test = self.create_sequences(test_array, self.config.window_size)

            mlflow.set_tracking_uri(self.config.mlflow_uri)
            mlflow.set_experiment("energy_forecasting_model_training")

            best_r2 = -np.inf
            best_model = None
            best_model_name = ""
            best_hp_values = {}

            with mlflow.start_run(run_name="model_trainer_hp_search"):
                for m_type in ["LSTM", "GRU"]:
                    model, r2, hp_values, val_loss, val_mae = self._tune_model_type(
                        m_type, X_train, y_train, X_test, y_test
                    )
                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model
                        best_model_name = m_type
                        best_hp_values = hp_values

                logging.info(f"Winner: {best_model_name} with R2: {best_r2:.4f}")
                mlflow.log_param("winning_model", best_model_name)
                mlflow.log_metric("winning_r2_score", best_r2)
                mlflow.log_params({f"winning_{k}": v for k, v in best_hp_values.items()})
                mlflow.tensorflow.log_model(best_model, artifact_path="best_model")

            model_path = os.path.join(self.config.root_dir, "model.h5")
            best_model.save(model_path)

            save_object(
                file_path=os.path.join(self.config.root_dir, "best_model_info.pkl"),
                obj={
                    "model_name": best_model_name,
                    "r2_score": best_r2,
                    "hyperparameters": best_hp_values,
                }
            )

        except Exception as e:
            raise CustomException(e, sys)