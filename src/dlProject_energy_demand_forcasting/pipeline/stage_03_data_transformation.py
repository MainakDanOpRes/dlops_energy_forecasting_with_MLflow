import os 
from pathlib import Path
import sys 
from src.dlProject_energy_demand_forcasting.config.configuration import ConfigurationManager
from src.dlProject_energy_demand_forcasting.components.data_transformation import DataTransformation
from src.dlProject_energy_demand_forcasting.utils.logger import logging
from src.dlProject_energy_demand_forcasting.utils.exception import CustomException


STAGE_NAME = "Data Transformation Stage"

class DataTransformationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            with open(Path("artifacts/data_validation/status.txt"), "r") as f:
                status = f.read().split(" ")[-1]

            if status == "True":
                config = ConfigurationManager()
                data_transformation_config = config.get_data_transformation_config()
                data_transformation = DataTransformation(config=data_transformation_config)
                data_transformation.train_test_spliting()
                data_transformation.initiate_data_transformation()
            else:
                raise Exception("You data schema is not valid")
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    try:
        obj = DataTransformationPipeline()
        obj.main()
    except Exception as e:
        raise CustomException(e, sys)