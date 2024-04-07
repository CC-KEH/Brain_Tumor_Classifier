from brain_tumor_classifier.utils.common import logger, load_binary_file
from brain_tumor_classifier.pipelines.stage_04_model_evaluation import Model_Evaluation_Training_Pipeline
from pathlib import Path


class PredictionPipeline:
    def __init__(self):
        self.model = load_binary_file(
            Path('artifacts/model_evaluation/best_model.h5'))

    def predict(self, data):
        prediction = self.model.predict(data)
        return prediction
