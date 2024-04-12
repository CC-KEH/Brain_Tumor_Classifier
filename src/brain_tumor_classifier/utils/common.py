import os
import shutil
import yaml
import json
from brain_tumor_classifier.utils import logger
from box import ConfigBox
from ensure import ensure_annotations
from box.exceptions import BoxValueError
import joblib
from typing import Any
from pathlib import Path
import pickle


def copy_images(source_folder, destination_folder):
    for root, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png') or file.endswith('.gif'):
                relative_dir = os.path.relpath(root, source_folder)
                destination_dir = os.path.join(
                    destination_folder, relative_dir)
                os.makedirs(destination_dir, exist_ok=True)
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_dir, file)
                shutil.copyfile(source_file_path, destination_file_path)


@ensure_annotations
def read_yaml(file_path: Path) -> ConfigBox:
    try:
        with open(file_path, 'r', encoding='utf8') as file:
            yaml_data = yaml.safe_load(file)
            logger.info(f'Yaml file loaded from: {file_path}')
            return ConfigBox(yaml_data)
    except BoxValueError:
        raise ValueError('Yaml File is Empty')
    except Exception as e:
        raise e


@ensure_annotations
def read_json(file_path) -> ConfigBox:
    with open(file_path, 'r') as file:
        json_data = json.load(file)
        logger.info(f'Json file loaded from: {file_path}')
        return ConfigBox(json_data)


def create_directories(path_to_dir: list, verbose=True):
    for path in path_to_dir:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f'Created directory at: {path}')


@ensure_annotations
def save_json(data: Any, file_path: Path):
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
    logger.info(f'Saved Json file at {file_path}')


@ensure_annotations
def save_binary_file(data: Any, file_path: Path):
    joblib.dump(data, file_path)
    logger.info(f'Binary file saved at {file_path}')


@ensure_annotations
def load_binary_file(file_path: Path) -> Any:
    data = joblib.load(file_path)
    logger.info(f'Successfully loaded binary file from: {file_path}')
    return data


@ensure_annotations
def get_size(path: Path) -> str:
    ''' Get Size in KB
    Args:
        path: path.
    Returns:
        str: size.
    '''
    size_in_kb = round(os.path.getsize(path)/1024)
    return f'~ {size_in_kb} KB'
