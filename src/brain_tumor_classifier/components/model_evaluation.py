import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from brain_tumor_classifier.utils.common import load_binary_file, logger
from brain_tumor_classifier.entity.config_entity import Model_Evaluation_Config
import os


class Model_Evaluation:
    def __init__(self, model_evaluation_config: Model_Evaluation_Config):
        self.config = model_evaluation_config

    def evaluate_models(self, models):
        best_model = models[0]
        best_model_accuracy = models[0].history['val_accuracy']
        for i, model in enumerate(models):
            # Plot and save loss
            plt.plot(model.history['loss'], label='Training Loss')
            plt.plot(model.history['val_loss'], label='Validation Loss')
            plt.legend()
            plt.savefig(
                f'{self.config.metric_file}/model_{i}_Loss_Val_Loss.png')
            plt.close()

            # Plot and save accuracy
            plt.plot(model.history['accuracy'], label='Training Accuracy')
            plt.plot(model.history['val_accuracy'],
                     label='Validation Accuracy')
            plt.legend()
            plt.savefig(
                f'{self.config.metric_file}/model_{i}_Accuracy_Val_Accuracy.png')
            plt.close()

            # Update best model if necessary
            if model.history['val_accuracy'] > best_model_accuracy:
                best_model_accuracy = model.history['val_accuracy']
                best_model = model

        return best_model, best_model_accuracy

    def initiate_model_evaluation(self):
        resnet_model = load_binary_file(os.path.join(
            self.config.model_path, 'resnet_model.h5'))
        vgg_model = load_binary_file(os.path.join(
            self.config.model_path, 'vgg_model.h5'))

        best_model, best_model_accuracy = self.evaluate_models(
            models=[vgg_model, resnet_model])
        logger.info('Finished Evaluating Models, Outputs are saved in CWD')
        logger.info(
            f'Best Model found: {best_model} with Validation Accuracy {best_model_accuracy}')
        best_model.save(os.path.join(
            self.config.best_model_path, 'best_model.h5'))
        return (best_model, best_model_accuracy)
