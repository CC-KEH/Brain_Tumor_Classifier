import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, image
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from tensorflow.keras.optimizers import Adagrad, SGD, Adam
from tensorflow.keras import regularizers
from tensorflow.keras.utils import img_to_array, load_img
from tensorflow.keras.layers import Sequential, Input, Flatten, Dense, Dropout, BatchNormalization
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from src.entity.config_entity import Model_Trainer_Config
import matplotlib.pyplot as plt
from src.utils.common import logger


class Model_Trainer:
    def __init__(self, model_trainer_config: Model_Trainer_Config, model_trainer_params):
        self.config = model_trainer_config
        self.params = model_trainer_params
        input_shape = tuple(
            self.params['VGG16']['input_tensor']['input_shape'])
        self.vgg_base_model = VGG16(input_tensor=Input(input_shape=input_shape),
                                    weights=self.params['VGG16']['weights'],
                                    include_top=self.params['VGG16']['include_top'])
        self.resnet_base_model = ResNet50(input_tensor=Input(input_shape=input_shape),
                                          weights=self.params['ResNet50']['weights'],
                                          include_top=self.params['ResNet50']['include_top'])

    def get_models(self):
        for layer in self.model_trainer_config.vgg_base_model.layers:
            layer.trainable = False

        for layer in self.model_trainer_config.resnet_base_model.layers:
            layer.trainable = False

        vgg_based_model = Sequential()
        vgg_based_model.add(self.model_trainer_config.vgg_base_model)
        vgg_based_model.add(Flatten())
        vgg_based_model.add(Dense(256, activation='relu'))
        vgg_based_model.add(Dense(4, activation='softmax'))

        resnet_based_model = Sequential()
        resnet_based_model.add(self.model_trainer_config.resnet_base_model)
        resnet_based_model.add(Flatten())
        resnet_based_model.add(Dense(256, activation='relu'))
        resnet_based_model.add(Dense(4, activation='softmax'))

        vgg_based_model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        resnet_based_model.compile(
            loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        return (vgg_based_model, resnet_based_model)

    def initiate_model_trainer(self, training_set, testing_set):
        vgg_model, resnet_model = self.get_models()

        vgg_model.fit(training_set, epochs=self.params['VGG16']['epochs'], validation_set=testing_set, steps_per_epoch=len(
            training_set), validation_steps=len(testing_set))
        logger.info('Finished Training VGG16 Model')

        resnet_model.fit(training_set, epochs=self.params['ResNet50']['epochs'], validation_set=testing_set, steps_per_epoch=len(
            training_set), validation_steps=len(testing_set))
        logger.info('Finished Training ResNet50 Model')

        vgg_model.save(os.path.join(self.config.model_path, 'vgg_model.h5'))
        resnet_model.save(os.path.join(
            self.config.model_path, 'resnet_model.h5'))
        logger.info(f'Saved both models at: {self.config.model_path}')
