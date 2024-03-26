import setuptools
from setuptools import setup, find_packages
from typing import List

with open('README.md', 'r') as f:
    long_description = f.read()

__version__ = "0.0.0"
REPO_NAME = 'Brain_Tumor_Classifier'
SRC_REPO = 'src'
AUTHOR_NAME = 'CC-KEH'
AUTHOR_EMAIL = 'example@example.com'

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_NAME,
    author_email=AUTHOR_EMAIL,
    description='Brain Tumor Classifier',
    long_description=long_description,
    url=f"https/github.com/{AUTHOR_NAME}/{REPO_NAME}",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where='src')
)
