from dataclasses import dataclass
import os,sys
from src.exception import CustomException
from src.logger import logging
from src.utils import MainUtils
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img,image
from tensorflow.keras.applications import VGG16,ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adagrad,SGD,Adam
from tensorflow.keras import regularizers
from tensorflow.keras.utils import img_to_array,load_img
from tensorflow.keras.layers import Sequential,Input,Flatten,Dense,Dropout,BatchNormalization
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

TENSOR_SIZE = (224,224,3)

@dataclass
class Model_Trainer_Config:
    preprocessor_path = os.path.join('artifacts','model.h5')
    vgg_base_model = VGG16(input_tensor = Input(input_shape=TENSOR_SIZE),weights='imagenet',include_top=False)
    resnet_base_model = ResNet50(input_tensor = Input(input_shape=TENSOR_SIZE),weights='imagenet',include_top=False)
    
class Model_Trainer:
    def __init__(self):
        self.utils = MainUtils()
        self.model_trainer_config = Model_Trainer_Config()
        
    def get_models(self):
        for layer in self.model_trainer_config.vgg_base_model.layers:
            layer.trainable = False
        
        for layer in self.model_trainer_config.resnet_base_model.layers:
            layer.trainable = False
            
        vgg_based_model = Sequential()    
        vgg_based_model.add(self.model_trainer_config.vgg_base_model)    
        vgg_based_model.add(Flatten())    
        vgg_based_model.add(Dense(256,activation='relu'))    
        vgg_based_model.add(Dense(4,activation='softmax'))
        
        resnet_based_model = Sequential()
        resnet_based_model.add(self.model_trainer_config.resnet_base_model)
        resnet_based_model.add(Flatten())
        resnet_based_model.add(Dense(256,activation='relu'))
        resnet_based_model.add(Dense(4,activation='softmax'))    

        vgg_based_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        resnet_based_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        
        return (vgg_based_model,resnet_based_model)
    
    def evaluate_models(models):
        best_model = models[0]
        best_model_accuracy = models[0].history['val_accuracy']
        for model in models:
            plt.plot(model.history['loss'],label='Training_Loss')
            plt.plot(model.history['val_loss'],label='Validation_Loss')
            plt.legend()
            plt.savefig(f'{model} Loss_Val_Loss')
        
        for model in models:
            if(model.history['val_accuracy']>best_model_accuracy):
                best_model_accuracy = model.history['val_accuracy']
                best_model = model
            plt.plot(model.history['accuracy'],label='Training_Accuracy')
            plt.plot(model.history['val_accuracy'],label='Validation_Accuracy')
            plt.legend()
            plt.savefig(f'{model} Accuracy_Val_Accuracy')

        return best_model,best_model_accuracy
             
    def initiate_model_trainer(self,training_set,testing_set):
        # Read Dataset
        vgg_model,resnet_model = self.get_models()
        vgg_model.fit(training_set,epochs=10,validation_set=testing_set,steps_per_epoch=len(training_set),validation_steps=len(testing_set))
        logging.info('Finished Training VGG16 Model, Saving now')
        vgg_model.save('vgg_model.h5')
        resnet_model.fit(training_set,epochs=10,validation_set=testing_set,steps_per_epoch=len(training_set),validation_steps=len(testing_set))
        logging.info('Finished Training ResNet50 Model, Saving now')
        resnet_model.save('resnet_model.h5')
        best_model,best_model_accuracy = self.evaluate_models(models=[vgg_model,resnet_model])
        logging.info('Finished Evaluating Models, Outputs are saved in CWD')
        print('Finished Evaluating Models, Outputs are saved in CWD')
        print(f'Best Model found: {best_model} with Validation Accuracy {best_model_accuracy}')
        return (best_model,best_model_accuracy)
        