import os
import urllib.request as request
import zipfile
from brain_tumor_classifier.utils.common import logger
from brain_tumor_classifier.utils import logger
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import pandas as pd
from brain_tumor_classifier.entity.config_entity import Data_Transformation_Config


class Data_Transformation:
    def __init__(self, config: Data_Transformation_Config):
        self.config = config

    def initiate_data_transformation(self):
        # Read Dataset
        IMAGE_SIZE = (224, 224)
        train_data_generator = ImageDataGenerator(
            rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        test_data_generator = ImageDataGenerator(rescale=1./255)
        training_set = train_data_generator.flow_from_directory(os.path.join(
            self.config.data_path, 'Training'), target_size=IMAGE_SIZE, batch_size=32, class_mode='categorical')
        testing_set = test_data_generator.flow_from_directory(os.path.join(
            self.config.data_path, 'Testing'), target_size=IMAGE_SIZE, batch_size=32, class_mode='categorical')

        logger.info(
            f'Successfully applied Transformation techniques on data')
        return (training_set, testing_set)
