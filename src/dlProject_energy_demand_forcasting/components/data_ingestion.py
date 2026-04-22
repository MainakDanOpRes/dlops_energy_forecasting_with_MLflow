import os 
import urllib.request as request
import zipfile
from src.dlProject_energy_demand_forcasting.utils.logger import logging
from src.dlProject_energy_demand_forcasting.utils.utils import get_size
from src.dlProject_energy_demand_forcasting.entity.config_entity import DataIngestionConfig
from pathlib import Path


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_url,
                filename=self.config.local_data_file
            )
            logging.info(f"{filename} downloaded with the following info: \n{headers}")
        else:
            logging.info(f"File already exists of size: {get_size(Path(self.config.local_data_file))}")
        if not zipfile.is_zipfile(self.config.local_data_file):
            with open(self.config.local_data_file, 'rb') as f:
                preview = f.read(300)
            raise ValueError(
                f"Downloaded file is not a valid zip.\n"
                f"File size: {os.path.getsize(self.config.local_data_file)} bytes\n"
                f"Preview: {preview}"
            )
    
    def extract_zip_file(self):
        
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)