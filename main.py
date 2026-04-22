import os 
import sys 
from src.dlProject_energy_demand_forcasting.utils.logger import logging
from src.dlProject_energy_demand_forcasting.utils.exception import CustomException
from src.dlProject_energy_demand_forcasting.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from src.dlProject_energy_demand_forcasting.pipeline.stage_02_data_validation import DataValidationPipeline

try:
    obj = DataIngestionPipeline()
    obj.main()
except Exception as e:
    raise CustomException(e, sys)


try:
    obj = DataValidationPipeline()
    obj.main()
except Exception as e:
    raise CustomException(e, sys)