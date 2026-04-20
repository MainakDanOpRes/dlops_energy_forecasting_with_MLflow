import os
import sys
from src.dlProject_energy_demand_forcasting.utils.exception import CustomException
from src.dlProject_energy_demand_forcasting.utils.logger import logging
import pandas as pd
from dataclasses import dataclass


@dataclass
class DLDataIngestionConfig:
    raw_data_path: str = 