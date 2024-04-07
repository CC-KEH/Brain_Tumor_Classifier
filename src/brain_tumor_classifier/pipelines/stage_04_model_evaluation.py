import sys
import os
from brain_tumor_classifier.utils.common import logger
from brain_tumor_classifier.components.model_evaluation import Model_Evaluation
from brain_tumor_classifier.config.configuration import Configuration_Manager

STAGE_NAME = "Model Evaluation Stage"


class Model_Evaluation_Training_Pipeline():
    def __init__(self):
        pass

    def main(self):
        config = Configuration_Manager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = Model_Evaluation(config=model_evaluation_config)
        model_evaluation.initiate_model_evaluation()


if __name__ == "__main__":
    try:
        logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
        obj = Model_Evaluation_Training_Pipeline()
        obj.main()
        logger.info(
            f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx=============x')
    except Exception as e:
        logger.exception(e)
        raise e
