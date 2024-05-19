import os
from src.brain_tumor_classifier.utils.common import get_size, logger
from src.brain_tumor_classifier.entity.config_entity import Data_Ingestion_Config
from pathlib import Path
from src.brain_tumor_classifier.utils.common import copy_images


class Data_Ingestion:
    def __init__(self, config: Data_Ingestion_Config):
        self.config = config

    def copy_data(self):
        os.makedirs(self.config.local_data_path, exist_ok=True)
        copy_images(self.config.source_path, self.config.local_data_path)
        logger.info(f'Successfully saved data')

    def initiate_data_ingestion(self):
        self.copy_data()
