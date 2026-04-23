import os 
import sys 
from src.dlProject_energy_demand_forcasting.config.configuration import ConfigurationManager
from src.dlProject_energy_demand_forcasting.components.data_validation import DataValidation
from src.dlProject_energy_demand_forcasting.utils.logger import logging
from src.dlProject_energy_demand_forcasting.utils.exception import CustomException


STAGE_NAME = "Data Validation Stage"

class DataValidationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logging.info(f" {STAGE_NAME} started...")
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            data_validation = DataValidation(config=data_validation_config)
            data_validation.validate_all_columns()
            logging.info(f" {STAGE_NAME} completed!")

        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    try:
        obj = DataValidationPipeline()
        obj.main()
    except Exception as e:
        raise CustomException(e, sys)
