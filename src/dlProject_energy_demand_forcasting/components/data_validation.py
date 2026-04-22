import pandas as pd
import os
import sys
from src.dlProject_energy_demand_forcasting.utils.logger import logging
from src.dlProject_energy_demand_forcasting.utils.utils import get_size
from src.dlProject_energy_demand_forcasting.utils.exception import CustomException
from src.dlProject_energy_demand_forcasting.entity.config_entity import DataValidationConfig
from pathlib import Path

class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self) -> bool:
        try:
            logging.info("Data validation started...")
            output_dir = os.path.dirname(self.config.STATUS_FILE)
            os.makedirs(output_dir, exist_ok=True)

            validation_status = True 
            
            if not os.path.exists(self.config.unzip_data_dir):
                validation_status = False
                with open(self.config.STATUS_FILE, 'w') as f:
                    f.write(f"validation status: {validation_status} (Input file missing)")
                return validation_status

            data = pd.read_csv(self.config.unzip_data_dir, sep=';', na_values=['?'], low_memory=False)
            
            schema_dict = self.config.all_schema.get('columns', {})

            for col in data.columns:
                if col not in schema_dict:
                    validation_status = False
                    break
                
                expected_type = schema_dict[col]
                actual_type = str(data[col].dtype)

                if actual_type != expected_type:
                    validation_status = False
                    break

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"validation status: {validation_status}")
            logging.info("Data validation completed!")
            return validation_status

        except Exception as e:
            raise CustomException(e, sys)