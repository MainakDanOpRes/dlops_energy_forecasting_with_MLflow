import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

    def build_model(self, model_type, input_shape):
        """
        Builds the architecture (LSTM or GRU) based on configuration.
        """
        logging.info(f"Building {model_type} architecture...")
        p = self.config.all_model_params[model_type]
        
        model = Sequential([Input(shape=input_shape)])

        # Core RNN Layer Selection
        if model_type.upper() == "LSTM":
            model.add(LSTM(units=p.units_layer_1, return_sequences=True))
            model.add(Dropout(p.dropout_rate))
            model.add(LSTM(units=p.units_layer_2))
        
        elif model_type.upper() == "GRU":
            model.add(GRU(units=p.units_layer_1, return_sequences=True))
            model.add(Dropout(p.dropout_rate))
            model.add(GRU(units=p.units_layer_2))
        
        else:
            raise ValueError(f"Unsupported model type: {self.config.model_name}")

        # Output Head
        model.add(Dense(p.dense_units, activation='relu'))
        model.add(Dense(units=1))

        # Optimizer selection
        opt = tf.keras.optimizers.get(p.optimizer)
        opt.learning_rate = p.learning_rate
        model.compile(optimizer=opt, loss=p.loss, metrics=['mae'])
        
        return model

    def initiate_model_trainer(self):
        try:
            logging.info("Loading transformed train and test data.")
            train_df = pd.read_csv(self.config.train_data_path, index_col='Datetime')
            test_df = pd.read_csv(self.config.test_data_path, index_col='Datetime')

            # Extract the target column values (Expected to be scaled already)
            target = self.config.target_column
            train_array = train_df[target].values.reshape(-1, 1)
            test_array = test_df[target].values.reshape(-1, 1)

            logging.info(f"Creating 3D sequences with window size: {self.config.window_size}")
            X_train, y_train = self.create_sequences(train_array, self.config.window_size)
            X_test, y_test = self.create_sequences(test_array, self.config.window_size)

            best_r2 = -np.inf
            best_model = None
            best_model_name = ""

            for m_type in ["LSTM", "GRU"]:
                logging.info(f">>> Starting training for: {m_type}")
                p = self.config.all_model_params[m_type]
                
                model = self.build_model(m_type, (self.config.window_size, 1))
                
                model.fit(
                    X_train, y_train,
                    epochs=p.epochs,
                    batch_size=p.batch_size,
                    validation_data=(X_test, y_test),
                    verbose=0,
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
                )

                # Evaluate
                preds = model.predict(X_test)
                r2 = r2_score(y_test, preds)
                logging.info(f"{m_type} achieved R2 Score: {r2:.4f}")

                # 3. Compare and keep the best
                if r2 > best_r2:
                    best_r2 = r2
                    best_model = model
                    best_model_name = m_type

            # 4. Final Save
            logging.info(f"Winner: {best_model_name} with R2: {best_r2:.4f}")
            model_path = os.path.join(self.config.root_dir, "model.h5")
            best_model.save(model_path)
            
            save_object(
                file_path=os.path.join(self.config.root_dir, "best_model_info.pkl"),
                obj={"model_name": best_model_name, "r2_score": best_r2}
            )

        except Exception as e:
            raise CustomException(e, sys)
