import os
import urllib.request as request
import zipfile
from src.utils.common import logger, save_dataset
from src.utils import logger
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import pandas as pd
from src.entity.config_entity import Data_Transformation_Config


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
            self.config.data_path, 'train'), target_size=IMAGE_SIZE, batch_size=32, class_mode='categorical')
        testing_set = test_data_generator.flow_from_directory(os.path.join(
            self.config.data_path, 'test'), target_size=IMAGE_SIZE, batch_size=32, class_mode='categorical')
        save_dataset(training_set, self.config.train_path)
        save_dataset(testing_set, self.config.test_path)
        logger.info(
            f'Successfully applied Transformation techniques on data, and saved to: {self.config.root_dir}')
        return (training_set, testing_set)
