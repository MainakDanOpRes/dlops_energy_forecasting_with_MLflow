from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    transformed_train_file_path: Path
    transformed_test_file_path: Path
    preprocessor_obj_file_path: Path
    date_col: str = 'Date'
    time_col: str = 'Time'

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_data_path: Path
    test_data_path: Path
    all_model_params: dict  
    tuning_params: dict
    window_size: int
    target_column: str
    tuner_dir: Path
    mlflow_uri: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    preprocessor_path: Path   
    target_column: str
    window_size: int
    metric_file_name: Path
    mlflow_uri: str
    all_params: dict

    