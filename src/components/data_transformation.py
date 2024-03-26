import os
import urllib.request as request
import zipfile
from src.utils.common import get_size, logger
from src.utils import logger
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import numpy as np
import pandas as pd
from src.entity.config_entity import Data_Transformation_Config


class Data_Ingestion:
    def __init__(self, config: Data_Transformation_Config):
        self.config = config

    def initiate_data_transformation(self, train_path, test_path):
        # Read Dataset
        IMAGE_SIZE = (224, 224)
        train_data_generator = ImageDataGenerator(
            rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
        test_data_generator = ImageDataGenerator(rescale=1./255)
        training_set = train_data_generator.flow_from_directory(
            train_path, target_size=IMAGE_SIZE, batch_size=32, class_mode='categorical')
        testing_set = test_data_generator.flow_from_directory(
            test_path, target_size=IMAGE_SIZE, batch_size=32, class_mode='categorical')
        return (training_set, testing_set)
