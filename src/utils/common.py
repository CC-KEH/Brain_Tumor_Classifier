import os
import shutil
import yaml
import json
from src.utils import logger
from box import ConfigBox
from ensure import ensure_annotations
from box.exceptions import BoxValueError
import joblib
from typing import Any
from pathlib import Path


def copy_images(source_folder, destination_folder):
    # Walk through each directory in the source folder
    for root, dirs, files in os.walk(source_folder):
        # Iterate over files in the current directory
        for file in files:
            # Check if the file is an image (you can adjust this condition according to your image formats)
            if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png') or file.endswith('.gif'):
                # Get the relative path of the current directory relative to the source folder
                relative_dir = os.path.relpath(root, source_folder)
                # Construct the corresponding destination directory path
                destination_dir = os.path.join(
                    destination_folder, relative_dir)
                # Ensure the destination directory exists, create if it doesn't
                os.makedirs(destination_dir, exist_ok=True)
                # Construct the full path of the source and destination file
                source_file_path = os.path.join(root, file)
                destination_file_path = os.path.join(destination_dir, file)
                # Copy the file from source to destination
                shutil.copyfile(source_file_path, destination_file_path)
                logger.info(
                    f"Copied {source_file_path} to {destination_file_path}")


@ensure_annotations
def read_yaml(file_path: Path) -> ConfigBox:
    try:
        with open(file_path, encoding='utf-8') as file:
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
def save_dataset(dataset: Any, save_path: Path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for i, (images, labels) in enumerate(dataset):
        for j in range(len(images)):
            image = images[j]
            label = labels[j]
            image_path = os.path.join(
                save_path, str(i) + '_' + str(j) + '.jpg')
            label_dir = os.path.join(save_path, str(label))
            if not os.path.exists(label_dir):
                os.makedirs(label_dir)
            image.save(image_path)

        if i == len(dataset) - 1:
            break
