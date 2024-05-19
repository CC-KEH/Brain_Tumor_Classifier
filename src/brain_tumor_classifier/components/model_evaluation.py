import numpy as np
from sklearn.metrics import accuracy_score
from src.brain_tumor_classifier.utils.common import logger
from src.brain_tumor_classifier.entity.config_entity import Model_Evaluation_Config
import os

class Model_Evaluation:
    def __init__(self, model_evaluation_config: Model_Evaluation_Config):
        self.config = model_evaluation_config
        if not os.path.exists(self.config.metric_file):
            os.makedirs(self.config.metric_file)

    def evaluate_model(self, models, testing_set):
        vgg_model = models[0]

        preds_vgg = vgg_model.predict(testing_set)
        
        preds_vgg_classes = np.argmax(preds_vgg, axis=1)

        true_labels = testing_set.classes

        accuracy_vgg = accuracy_score(true_labels, preds_vgg_classes)

        logger.info(f'VGG16 Model Accuracy: {accuracy_vgg}')

        with open(os.path.join(self.config.metric_file, 'metrics.txt'), 'a') as f:
            f.write(f'Model: VGG16 with Validation Accuracy {accuracy_vgg}\n')
    
    def initiate_model_evaluation(self,models,testing_set):
        self.evaluate_model(models=models,testing_set=testing_set)
        logger.info('Successfully evaluated the models.')