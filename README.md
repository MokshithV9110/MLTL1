This project implements a Support Vector Machine (SVM) model to classify images of cats and dogs using the Kaggle *Dogs vs. Cats* dataset.

## Dataset

The dataset used for this project is the [Dogs vs. Cats Dataset](https://www.kaggle.com/c/dogs-vs-cats/data) from Kaggle. It consists of images of cats and dogs in two folders:

•⁠  ⁠⁠ train/ ⁠: Contains labeled images of cats and dogs.
•⁠  ⁠⁠ test/ ⁠: Contains images for testing the model.

Each image is resized to 64x64 pixels and converted to grayscale before training.

## Requirements

•⁠  ⁠Python 3.x
•⁠  ⁠Libraries: ⁠ numpy ⁠, ⁠ pandas ⁠, ⁠ scikit-learn ⁠, ⁠ opencv-python ⁠, ⁠ matplotlib ⁠, ⁠ joblib ⁠

Install the dependencies by running:

```bash
pip install numpy pandas scikit-learn opencv-python matplotlib joblib
