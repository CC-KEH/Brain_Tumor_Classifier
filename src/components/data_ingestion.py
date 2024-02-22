from dataclasses import dataclass
import os,sys
from src.exception import CustomException
from src.logger import logging
from src.utils import MainUtils
import numpy as np
import pandas as pd


@dataclass
class Data_Ingestion_Config:
    train_path = os.path.join('artifacts/train')
    test_path = os.path.join('artifacts/test')
    
class Data_Ingestion:
    def __init__(self):
        self.utils = MainUtils()
        self.data_ingestion_config = Data_Ingestion_Config()
        
    def initiate_data_ingestion(self):
        # No Idea what to do here at the moment
        pass
    
