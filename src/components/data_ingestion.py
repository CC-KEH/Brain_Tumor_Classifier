import os
import urllib.request as request
import zipfile
from src.utils.common import get_size, logger
from src.entity.config_entity import Data_Ingestion_Config
from pathlib import Path


class Data_Ingestion:
    def __init__(self, config: Data_Ingestion_Config):
        self.config = config

    def download_data(self):
        if not os.path.exists(self.config.local_data_path):
            filename, header = request.urlretrieve(
                url=self.config.source_URL, filename=self.config.local_data_path)
            logger.info(
                f'Successfully downloaded data {filename} with info: {header} at: {self.config.local_data_path}')
        else:
            logger.info(
                f'Data already exists of size {get_size(Path(self.config.local_data_path))}')

    def extract_data(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)

    def initiate_data_ingestion(self):
        self.download_data()
        self.extract_data()
