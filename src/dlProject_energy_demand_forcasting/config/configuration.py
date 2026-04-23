from src.dlProject_energy_demand_forcasting.constants import *
from src.dlProject_energy_demand_forcasting.utils.utils import read_yaml, create_directories
from src.dlProject_energy_demand_forcasting.entity.config_entity import (DataIngestionConfig, 
                                                                         DataValidationConfig, 
                                                                         DataTransformationConfig)

class ConfigurationManager:
    def __init__(self, config_filepath = CONFIG_FILE_PATH,
                 params_filepath = PARAMS_FILE_PATH,
                 schema_filepath = SCHEMA_FILE_PATH):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir
        )
        return data_ingestion_config
    
    def get_data_validation_config(self) -> DataValidationConfig:
        config = self.config.data_validation
        schema = self.schema.columns

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=config.root_dir,
            STATUS_FILE= config.STATUS_FILE,
            unzip_data_dir=config.unzip_data_dir,
            all_schema=schema
        ) 

        return data_validation_config
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        create_directories([config.root_dir])
        data_transformation_config = DataTransformationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            transformed_train_file_path = config.transformed_train_file_path,
            transformed_test_file_path = config.transformed_test_file_path,
            preprocessor_obj_file_path = config.preprocessor_obj_file_path,
            date_col = config.date_col,
            time_col = config.time_col,
        ) 
        return data_transformation_config