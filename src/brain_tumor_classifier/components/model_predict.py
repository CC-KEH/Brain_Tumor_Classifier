from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from src.brain_tumor_classifier.utils.common import logger
import numpy as np

class Model_Predict:
    def __init__(self):
        self.model = load_model('artifacts/model_trainer/vgg.h5')

    def predict(self, img):
        img=img/255
        img=np.expand_dims(img,axis=0)
        img_data=preprocess_input(img)
        prediction = np.argmax(self.model.predict(img_data), axis=1)
        return prediction[0]
    
    def initiate_prediction_pipeline(self,img_path):
        try:
            logger.info('>>>>>>> Prediction Pipeline Started <<<<<<<')
            img = image.load_img(img_path, target_size=(224,224))
            img = image.img_to_array(img)
            prediction = self.predict(image)
            logger.info('>>>>>>> Prediction Pipeline Completed <<<<<<<\n\nx============================x')
        except Exception as e:
            logger.exception(e)
            raise e
        return prediction