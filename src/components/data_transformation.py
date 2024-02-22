from dataclasses import dataclass
import os,sys
from src.exception import CustomException
from src.logger import logging
from src.utils import MainUtils
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

@dataclass
class Data_Transformation_Config:
    preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
    
class Data_Transformation:
    def __init__(self):
        self.utils = MainUtils()
        self.data_transformation_config = Data_Transformation_Config()
        
    def initiate_data_transformation(self,train_path,test_path):
        # Read Dataset
        IMAGE_SIZE = (224,224)
        train_data_generator = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
        test_data_generator = ImageDataGenerator(rescale=1./255)
        training_set = train_data_generator.flow_from_directory(train_path,target_size=IMAGE_SIZE,batch_size=32,class_mode='categorical')
        testing_set = test_data_generator.flow_from_directory(test_path,target_size=IMAGE_SIZE,batch_size=32,class_mode='categorical')
        return (training_set,testing_set)
