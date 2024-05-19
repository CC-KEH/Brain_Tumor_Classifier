import sys
import os
from src.brain_tumor_classifier.utils.common import logger
from src.brain_tumor_classifier.components.data_transformation import Data_Transformation
from src.brain_tumor_classifier.config.configuration import Configuration_Manager

STAGE_NAME = "Data Transformation Stage"


class Data_Transformation_Training_Pipeline():
    def __init__(self):
        pass

    def main(self):
        config = Configuration_Manager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = Data_Transformation(
            config=data_transformation_config)
        training_set,validation_set, testing_set = data_transformation.initiate_data_transformation()
        return (training_set, validation_set, testing_set)


if __name__ == "__main__":
    try:
        logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
        obj = Data_Transformation_Training_Pipeline()
        obj.main()
        logger.info(
            f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx=============x')
    except Exception as e:
        logger.exception(e)
        raise e
