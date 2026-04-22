import os 
import sys 
from src.dlProject_energy_demand_forcasting.config.configuration import ConfigurationManager
from src.dlProject_energy_demand_forcasting.components.data_ingestion import DataIngestion
from src.dlProject_energy_demand_forcasting.utils.logger import logging
from src.dlProject_energy_demand_forcasting.utils.exception import CustomException


STAGE_NAME = "Data Ingestion Stage"

class DataIngestionPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logging.info(f" {STAGE_NAME} started...")
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()
            logging.info(f" {STAGE_NAME} completed!")

        except Exception as e:
            raise CustomException(e, sys)
        
# if __name__ == '__main__':
#     try:
#         obj = DataIngestionPipeline()
#         obj.main()
#     except Exception as e:
#         raise CustomException(e, sys)
