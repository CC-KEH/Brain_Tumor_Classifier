import sys
import os
from src.utils.common import logger
from src.components.model_trainer import Model_Trainer
from src.configuration.configuration import Configuration_Manager

STAGE_NAME = "Model Trainer Stage"


class Model_Trainer_Training_Pipeline():
    def __init__(self):
        pass

    def main(self):
        config = Configuration_Manager()
        model_trainer_config, model_trainer_params = config.get_model_trainer_config()
        model_trainer = Model_Trainer(
            config=model_trainer_config, model_trainer_params=model_trainer_params)
        model_trainer.initiate_model_trainer()


if __name__ == "__main__":
    try:
        logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
        obj = Model_Trainer_Training_Pipeline()
        obj.main()
        logger.info(
            f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx=============x')
    except Exception as e:
        logger.exception(e)
        raise e
