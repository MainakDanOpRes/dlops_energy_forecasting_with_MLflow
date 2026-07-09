import sys 
from src.dlProject_energy_demand_forcasting.config.configuration import ConfigurationManager
from src.dlProject_energy_demand_forcasting.components.model_evaluation import ModelEvaluation
from src.dlProject_energy_demand_forcasting.utils.exception import CustomException

STAGE_NAME = "Model Training Stage"

class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            config = ConfigurationManager()
            model_evaluation_config = config.get_model_evaluation_config()
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            model_evaluation.initiate_model_evaluation()
        except Exception as e:
            raise CustomException(e, sys)
        
if __name__ == '__main__':
    try:
        obj = ModelEvaluationPipeline()
        obj.main()
    except Exception as e:
        raise CustomException(e, sys)