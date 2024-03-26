from src.utils.common import logger, load_binary_file
from src.pipelines.stage_01_data_ingestion import Data_Ingestion_Training_Pipeline
from src.pipelines.stage_02_data_transformation import Data_Transformation_Training_Pipeline
from src.pipelines.stage_03_model_trainer import Model_Trainer_Training_Pipeline
from src.pipelines.stage_04_model_evaluation import Model_Evaluation_Training_Pipeline
from pathlib import Path


class PredictionPipeline:
    def __init__(self):
        self.model = load_binary_file(
            Path('artifacts/model_trainer/resnet_model.h5'))

    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction
