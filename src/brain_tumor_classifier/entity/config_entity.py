from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Data_Ingestion_Config:
    root_dir: Path
    source_path: Path
    local_data_path: Path


@dataclass(frozen=True)
class Data_Transformation_Config:
    root_dir:  Path
    data_path: Path
    local_data_path: Path


@dataclass(frozen=True)
class Model_Trainer_Config:
    root_dir:  Path
    train_path: Path
    test_path: Path
    model_path: Path


@dataclass(frozen=True)
class Model_Evaluation_Config:
    root_dir: Path
    test_data_path: Path
    model_path: Path
    metric_file: Path
