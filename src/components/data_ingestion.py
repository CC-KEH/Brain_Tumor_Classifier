import os
import urllib.request as request
import zipfile
from src.utils.common import get_size, logger
from src.entity.config_entity import Data_Ingestion_Config


class Data_Ingestion:
    def __init__(self, config: Data_Ingestion_Config):
        self.config = config

    def initiate_data_ingestion(self):
        train_dir = os.path.join('dataset/Training/')
        test_dir = os.path.join('dataset/Testing/')
        self.utils.copy_images(
            source_folder=train_dir, destination_folder=self.data_ingestion_config.train_path)
        self.utils.copy_images(
            source_folder=test_dir, destination_folder=self.data_ingestion_config.test_path)
