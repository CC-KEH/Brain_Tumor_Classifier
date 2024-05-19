import sys
import os
from src.brain_tumor_classifier.utils.common import logger
from src.brain_tumor_classifier.components.model_trainer import Model_Trainer
from src.brain_tumor_classifier.config.configuration import Configuration_Manager

STAGE_NAME = "Model Trainer Stage"


class Model_Trainer_Training_Pipeline():
    def __init__(self):
        pass

    def main(self, training_set,validation_set):
        config = Configuration_Manager()
        model_trainer_config, model_trainer_params = config.get_model_trainer_config()
        model_trainer = Model_Trainer(
            model_trainer_config=model_trainer_config, model_trainer_params=model_trainer_params)
        models = model_trainer.initiate_model_trainer(training_set=training_set, validation_set=validation_set)
        return models

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
