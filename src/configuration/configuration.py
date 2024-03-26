from src.utils.common import read_yaml, create_directories
from src.constant import *
from src.entity.config_entity import *


class Configuration_Manager:
    def __init__(self, config_path=CONFIG_PATH, params_path=PARAMS_PATH, schema_path=SCHEMA_PATH):
        self.config = read_yaml(config_path)
        self.config = read_yaml(config_path)
        self.config = read_yaml(config_path)
        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> Data_Ingestion_Config:
        config = self.config.Data_Ingestion
        create_directories([config.root_dir])
        data_ingestion_config = Data_Ingestion_Config(
            root_dir=config.root_dir, source_URL=config.source_URL, local_data_path=config.loca_data_path, unzip_dir=config.unzip_dir)
        return data_ingestion_config

    def get_data_ingestion_config(self) -> Data_Transformation_Config:
        config = self.config.Data_Transformation
        create_directories([config.root_dir])
        data_transformation_config = Data_Transformation_Config(
            root_dir=config.root_dir, data_path=config.data_path, train_path=config.train_path, test_path=config.test_path)
        return data_transformation_config

    def get_model_trainer_config(self) -> Model_Trainer_Config:
        config = self.config.Model_Trainer
        create_directories([config.root_dir])
        model_trainer_config = Model_Trainer_Config(
            root_dir=config.root_dir, data_path=config.data_path, train_path=config.train_path, test_path=config.test_path)
        return model_trainer_config

    def get_model_evaluation_config(self) -> Model_Evaluation_Config:
        config = self.config.Model_Evaluation
        create_directories([config.root_dir])
        model_evaluation_config = Model_Evaluation_Config(
            root_dir=config.root_dir, data_path=config.data_path, train_path=config.train_path, test_path=config.test_path)
        return model_evaluation_config
