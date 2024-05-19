import sys
import os
from src.brain_tumor_classifier.utils.common import logger
from src.brain_tumor_classifier.components.data_ingestion import Data_Ingestion
from src.brain_tumor_classifier.config.configuration import Configuration_Manager

STAGE_NAME = "Data Ingestion Stage"


class Data_Ingestion_Training_Pipeline():
    def __init__(self):
        pass

    def main(self):
        config = Configuration_Manager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = Data_Ingestion(config=data_ingestion_config)
        data_ingestion.initiate_data_ingestion()


if __name__ == "__main__":
    try:
        logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
        obj = Data_Ingestion_Training_Pipeline()
        obj.main()
        logger.info(
            f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx=============x')
    except Exception as e:
        logger.exception(e)
        raise e
