import sys,os
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import Data_Ingestion
from src.components.data_transformation import Data_Transformation
from src.components.model_trainer import Model_Trainer

if __name__ =="__main__":
    data_ingestor = Data_Ingestion()
    train_path,test_path = data_ingestor.initiate_data_ingestion()
    
    data_transformer = Data_Transformation()
    training_set,testing_set = data_transformer.initiate_data_transformation(train_path,test_path)
    
    model_trainer = Model_Trainer()
    model_trainer.initiate_model_trainer(training_set,testing_set)
    