import sys
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

from src.dlProject_energy_demand_forcasting.utils.exception import CustomException
from src.dlProject_energy_demand_forcasting.utils.logger import logging
from src.dlProject_energy_demand_forcasting.utils.utils import save_object

from src.dlProject_energy_demand_forcasting.entity.config_entity import DataTransformationConfig

class TimeSeriesScaling(BaseEstimator, TransformerMixin):
    """
    Scales features using MinMaxScaler while preserving 
    the
    """

    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.columns = None
        self.index = None

    def fit(self, X, y=None):
        self.scaler.fit(X)
        self.columns = X.columns
        self.is_fitted_ = True
        return self
    
    def transform(self, X):
        X_scaled = self.scaler.transform(X)
        # Convert back to DataFrame to keep Datetime Index and Column Names
        return pd.DataFrame(X_scaled, columns=self.columns, index=X.index)



class DatetimeIndexer(BaseEstimator, TransformerMixin):
    """Converts Date and Time columns into a Datetime Index."""
    def __init__(self, date_col, time_col):
        self.date_col = date_col
        self.time_col = time_col

    def fit(self,X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed['Datetime'] = pd.to_datetime(
            X_transformed[self.date_col] + ' ' + X_transformed[self.time_col],
            format='%d/%m/%Y %H:%M:%S'
        )
        X_transformed.drop([self.date_col, self.time_col], axis=1, inplace=True)
        X_transformed.set_index('Datetime', inplace=True)
        return X_transformed


class TimeSeriesImputer(BaseEstimator, TransformerMixin):
    """Imputes missing values using time-based interpolation."""
    def __init__(self, method='time'):
        self.method = method

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        X_transformed = X.copy()
        X_transformed = X_transformed.interpolate(method=self.method)
        X_transformed.bfill(inplace=True)
        return X_transformed
    

class TimeSeriesResampler(BaseEstimator, TransformerMixin):
    """
    Resamples all numeric columns to hourly averages.
    The full wide DataFrame is preserved so every column is available
    for independent per-column model training or prediction later.
    """
    def __init__(self, frequency='h'):
        self.frequency = frequency

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X):
        # Cast every column to numeric, coercing any non-numeric leftovers to NaN
        X_numeric = X.apply(pd.to_numeric, errors='coerce')
        return X_numeric.resample(self.frequency).mean()
    
class DataTransformation:
    """
    Transforms raw data into a clean, hourly-resampled wide DataFrame that
    retains ALL numeric columns. One shared preprocessor.pkl is saved and
    reused by every per-column model at training and prediction time.
    """
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def train_test_spliting(self):
        logging.info("initiated train test split...")
        try:
            data = data = pd.read_csv(self.config.data_path, sep=';', na_values=['?'], low_memory=False)
            self.train, self.test = train_test_split(data, shuffle=False)

            self.train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False, header = True)
            self.test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False, header = True)

            logging.info("train test split completed!")
            
        except Exception as e:
            raise CustomException(e, sys)

    def get_data_transformer_object(self):
        logging.info("Initializing the data transformation pipeline (all columns).")
        try:
            pipeline = Pipeline(steps=[
                ("indexer", DatetimeIndexer(
                    date_col=self.config.date_col,
                    time_col=self.config.time_col
                )),
                ("imputer", TimeSeriesImputer(method='time')),
                ("resampler", TimeSeriesResampler(frequency='h')),
                ("scaler", TimeSeriesScaling())
            ])
            return pipeline
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        try:
            # train_df = pd.read_csv(train_path, low_memory=False)
            # test_df = pd.read_csv(test_path, low_memory=False)
            logging.info("Read train and test data completed.")

            preprocessing_obj = self.get_data_transformer_object()

            logging.info("Applying preprocessing pipeline — all numeric columns retained.")
            train_processed = preprocessing_obj.fit_transform(self.train)
            test_processed = preprocessing_obj.transform(self.test)

            logging.info(f"Columns available after transformation: {list(train_processed.columns)}")

            train_processed.to_csv(
                os.path.join(self.config.root_dir, "train_transformed.csv"),
                index=True, header=True
            )
            test_processed.to_csv(
                os.path.join(self.config.root_dir, "test_transformed.csv"),
                index=True, header=True
            )
            logging.info('Saved transformed datasets to artifacts.')

            save_object(
                file_path=self.config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Saved shared preprocessor.pkl.')

            # return (
            #     self.config.transformed_train_file_path,
            #     self.config.transformed_test_file_path,
            #     self.config.preprocessor_obj_file_path
            # )

        except Exception as e:
            raise CustomException(e, sys)
