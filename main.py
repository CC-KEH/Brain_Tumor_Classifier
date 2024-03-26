from src.utils import logger
from src.pipelines.stage_01_data_ingestion import Data_Ingestion_Training_Pipeline
from src.pipelines.stage_02_data_transformation import Data_Transformation_Training_Pipeline
from src.pipelines.stage_03_model_trainer import Model_Trainer_Training_Pipeline
from src.pipelines.stage_04_model_evaluation import Model_Evaluation_Training_Pipeline

STAGE_NAME = 'Data Ingestion Stage'

try:
    logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
    obj = Data_Ingestion_Training_Pipeline()
    obj.main()
    logger.info(
        f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx============================x')
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = 'Data Transformation'

try:
    logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
    obj = Data_Transformation_Training_Pipeline()
    obj.main()
    logger.info(
        f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx============================x')
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = 'Model Trainer Stage'

try:
    logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
    obj = Model_Trainer_Training_Pipeline()
    obj.main()
    logger.info(
        f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx============================x')
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = 'Model Evaluation Stage'

try:
    logger.info(f'>>>>>>> STAGE {STAGE_NAME} Started <<<<<<<')
    obj = Model_Evaluation_Training_Pipeline()
    obj.main()
    logger.info(
        f'>>>>>>> STAGE {STAGE_NAME} Completed <<<<<<<\n\nx=============x')
except Exception as e:
    logger.exception(e)
    raise e
