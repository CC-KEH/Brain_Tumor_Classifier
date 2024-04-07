from brain_tumor_classifier.utils.common import read_yaml, create_directories
from brain_tumor_classifier.constant import *
from brain_tumor_classifier.entity.config_entity import *
import os


class Configuration_Manager:
    def __init__(self, config_path=CONFIG_PATH, params_path=PARAMS_PATH, schema_path=SCHEMA_PATH):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        self.schema = read_yaml(schema_path)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> Data_Ingestion_Config:
        config = self.config.Data_Ingestion
        create_directories([config.root_dir])
        data_ingestion_config = Data_Ingestion_Config(
            root_dir=config.root_dir, source_path=config.source_path, local_data_path=config.local_data_path)
        return data_ingestion_config

    def get_data_transformation_config(self) -> Data_Transformation_Config:
        config = self.config.Data_Transformation
        create_directories([config.root_dir])
        data_transformation_config = Data_Transformation_Config(
            root_dir=config.root_dir, data_path=config.data_path, local_data_path=config.local_data_path)
        return data_transformation_config

    def get_model_trainer_config(self) -> Model_Trainer_Config:
        config = self.config.Model_Trainer
        params = self.params
        create_directories([config.root_dir])
        model_trainer_config = Model_Trainer_Config(
            root_dir=config.root_dir, train_path=config.train_path, test_path=config.test_path, model_path=config.model_path)
        return model_trainer_config, params

    def get_model_evaluation_config(self) -> Model_Evaluation_Config:
        config = self.config.Model_Evaluation
        create_directories([config.root_dir])
        model_evaluation_config = Model_Evaluation_Config(
            root_dir=config.root_dir, test_data_path=config.test_data_path, model_path=config.model_path, metric_file=config.metric_file)
        return model_evaluation_config
