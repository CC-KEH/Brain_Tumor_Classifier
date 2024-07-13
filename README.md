# Brain Tumor Classifier

## Project Overview

The Brain Tumor Classifier project aims to classify brain MRI images into four categories: glioma, meningioma, no tumor, and pituitary tumor. This project leverages a pre-trained VGG model to achieve high accuracy in classification. The goal is to aid in the early detection and diagnosis of brain tumors, improving treatment outcomes.

## Data Description

The dataset used in this project contains MRI images categorized into the following classes:

- **Glioma**: A type of tumor that occurs in the brain and spinal cord.

- **Meningioma**: A tumor that arises from the meninges, the membranes that surround the brain and spinal cord.

- **No Tumor**: Images with no signs of a tumor.

- **Pituitary**: A tumor that occurs in the pituitary gland.

## Installation

To run this project, you'll need to have Python installed along with several packages. You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage

To use the model for classifying brain MRI images, follow these steps:

1. Clone the repository:

    ```bash
    git clone https://github.com/CC-KEH/Brain_Tumor_Classifier.git
    ```

2. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the classification script:

    ```bash
    python application.py
    ```

## Modeling

This project utilizes the VGG model for image classification. The main steps involved in the modeling process are:

1. **Data Preprocessing**: Resizing images, normalizing pixel values, and augmenting the dataset to improve model robustness.
2. **Model Architecture**: Using a pre-trained VGG model with modifications to the final layers to suit the classification task.
3. **Training**: Fine-tuning the model on the brain MRI dataset to achieve optimal performance.
4. **Evaluation**: Assessing the model's performance using metrics such as accuracy, precision, recall, and F1-score.

## Contributing

Contributions to this project are welcome! If you have any suggestions or improvements, please open an issue or create a pull request.
