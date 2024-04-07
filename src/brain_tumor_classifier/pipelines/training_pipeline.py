from brain_tumor_classifier.utils.common import logger
from brain_tumor_classifier.components.data_ingestion import Data_Ingestion
from brain_tumor_classifier.components.data_transformation import Data_Transformation
from brain_tumor_classifier.components.model_trainer import Model_Trainer
from brain_tumor_classifier.components.model_evaluation import Model_Evaluation
from brain_tumor_classifier.components.model_trainer import Model_Trainer
from brain_tumor_classifier.config.configuration import Configuration_Manager

if __name__ == "__main__":
    try:
        logger.info('Data Ingestion Started \n')
        config = Configuration_Manager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = Data_Ingestion(data_ingestion_config)
        data_ingestion.initiate_data_ingestion()
    except Exception as e:
        logger.info('Data Ingestion Failed \n')
        raise e
    logger.info('Data Ingestion Completed \n')
    logger.info(
        '------------------------------------------------------------------ \n\n')

    logger.info('Data Transformation Started \n')
    try:
        config = Configuration_Manager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = Data_Transformation(data_transformation_config)
        data_transformation.initiate_data_transformation()
    except Exception as e:
        logger.info('Data Transformation Failed \n')
        raise e
    logger.info('Data Transformation Completed \n')
    logger.info(
        '------------------------------------------------------------------ \n\n')

    logger.info('Model Training Started \n')
    try:
        config = Configuration_Manager()
        model_trainer_config, model_trainer_params = config.get_model_trainer_config()
        model_trainer = Model_Trainer(
            model_trainer_config, model_trainer_params)
        model_trainer.initiate_model_trainer()
    except Exception as e:
        logger.info('Model Training Failed \n')
        raise e
    logger.info('Model Training Completed \n')
    logger.info(
        '------------------------------------------------------------------ \n\n')

    logger.info('Model Evaluation Started \n')
    try:
        config = Configuration_Manager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = Model_Evaluation(model_evaluation_config)
        model_evaluation.initiate_model_evaluation()
    except Exception as e:
        logger.info('Data Evaluation Failed \n')
        raise e
    logger.info('Data Evaluation Completed \n')
    logger.info(
        '------------------------------------------------------------------ \n\n')

    logger.info(
        'xxxxxxxxxxxxxxxxxxxxx<< END OF TRAINING PIPELINE >>xxxxxxxxxxxxxxxxxxxxx')
