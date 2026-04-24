import os 
from pathlib import Path
import sys 
from src.dlProject_energy_demand_forcasting.config.configuration import ConfigurationManager
from src.dlProject_energy_demand_forcasting.components.model_trainer import ModelTrainer
from src.dlProject_energy_demand_forcasting.utils.logger import logging
from src.dlProject_energy_demand_forcasting.utils.exception import CustomException


STAGE_NAME = "Model Training Stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config = model_trainer_config)
            model_trainer.initiate_model_trainer()
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    try:
        obj = ModelTrainingPipeline()
        obj.main()
    except Exception as e:
        raise CustomException(e, sys)