from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential
import keras.saving
from src.brain_tumor_classifier.entity.config_entity import Model_Trainer_Config
from src.brain_tumor_classifier.utils.common import logger
import keras
class Model_Trainer:
    def __init__(self, model_trainer_config: Model_Trainer_Config, model_trainer_params):
        self.config = model_trainer_config
        self.params = model_trainer_params
        input_shape = (224, 224, 3)
        self.vgg_base_model = VGG16(input_tensor=Input(shape=input_shape),
                                    weights=self.params['VGG16']['weights'],
                                    pooling=self.params['VGG16']['pooling'],
                                    include_top=self.params['VGG16']['include_top']
                                    )

    def get_model(self):
        for layer in self.vgg_base_model.layers:
            layer.trainable = False

        vgg_based_model = Sequential()
        vgg_based_model.add(self.vgg_base_model)
        vgg_based_model.add(Flatten())
        vgg_based_model.add(Dense(256, activation='relu'))
        vgg_based_model.add(Dense(4, activation='softmax'))

        vgg_based_model.compile(
            loss=self.params['VGG16']['loss'], 
            optimizer=self.params['VGG16']['optimizer'], 
            metrics=['accuracy']
            )

        return vgg_based_model


    def train_model(self, training_set,validation_set):
        vgg_model = self.get_model()

        callback = keras.callbacks.EarlyStopping(monitor='loss', patience=5)

        # Train VGG16 model
        vgg_model.fit(training_set, 
                      epochs=self.params['VGG16']['epochs'],
                      validation_data=validation_set, 
                      steps_per_epoch=len(training_set),
                      validation_steps=len(validation_set),
                    # callbacks=[callback]
                      )
        
        logger.info('Finished Training VGG16 Model')
        logger.info('Saving VGG16 Model')
        vgg_model.save(self.config.model_path+'/vgg16.h5')
        # vgg_model.save(self.config.model_path+'/vgg16.keras')
        
        logger.info(f'Saved model at: {self.config.model_path}')
        return vgg_model
    
    def initiate_model_trainer(self, training_set,validation_set):
        model = self.train_model(training_set,validation_set)
        return [model]