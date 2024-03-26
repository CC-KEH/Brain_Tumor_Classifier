import os
import shutil
import yaml
import json
from src.utils import logger


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
                logging.info(
                    f"Copied {source_file_path} to {destination_file_path}")


def read_yaml(file_path):
    pass


def create_directories(path_to_dir: list):
    pass


def read_json(file_path):
    pass
