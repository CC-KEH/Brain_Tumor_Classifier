import sys
import os
from src.logger import logging
from src.components.data_ingestion import Data_Ingestion
from src.components.data_transformation import Data_Transformation
from src.components.model_trainer import Model_Trainer
from src.components.model_evaluation import Model_Evaluation
from src.components.model_trainer import Model_Trainer
from src.configuration.configuration import Configuration_Manager

if __name__ == "__main__":
    try:
        logging.info('Data Ingestion Started \n')
        config = Configuration_Manager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = Data_Ingestion(data_ingestion_config)
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info('Data Ingestion Failed \n')
        raise e
    logging.info('Data Ingestion Completed \n')
    logging.info(
        '------------------------------------------------------------------ \n\n')

    logging.info('Data Transformation Started \n')
    try:
        config = Configuration_Manager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = Data_Ingestion(data_ingestion_config)
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logging.info('Data Transformation Failed \n')
        raise e
    logging.info('Data Transformation Completed \n')
    logging.info(
        '------------------------------------------------------------------ \n\n')

    logging.info('Model Training Started \n')
    try:
        config = Configuration_Manager()
        model_trainer_config = config.get_model_trainer_config()
        model_trainer = Model_Trainer(model_trainer_config)
        model_trainer.initiate_model_trainer()
    except Exception as e:
        logging.info('Model Training Failed \n')
        raise e
    logging.info('Model Training Completed \n')
    logging.info(
        '------------------------------------------------------------------ \n\n')

    logging.info('Model Evaluation Started \n')
    try:
        config = Configuration_Manager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = Model_Evaluation(model_evaluation_config)
        model_evaluation.initiate_model_evaluation()
    except Exception as e:
        logging.info('Data Evaluation Failed \n')
        raise e
    logging.info('Data Evaluation Completed \n')
    logging.info(
        '------------------------------------------------------------------ \n\n')

    logging.info(
        'xxxxxxxxxxxxxxxxxxxxx<< END OF TRAINING PIPELINE >>xxxxxxxxxxxxxxxxxxxxx')
