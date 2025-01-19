# Machine-Learning-for-Cat-and-Dog-Classification
This repository demonstrates a complete workflow for classifying cat and dog images using machine learning. It covers data preprocessing, training models (SVM, Random Forest, Logistic Regression, CNN, and K-means), and building a Flask-based web application with an interactive frontend for image upload and classification.

## Objective
To classify images as cats or dogs using machine learning models.

## Requirements
- Python 3.8+
- Flask
- OpenCV
- TensorFlow/Keras
- Scikit-learn
- Matplotlib
- Pandas
- NumPy

## Repository Structure
```
Machine-Learning-for-Cat-and-Dog-Classification/
|-- animals/
|   |-- cat/
|   |-- dog/
|-- datasets/
|-- saved_models/
|   |-- cnn_model.h5
|   |-- kmeans.pkl
|   |-- logistic_regression.pkl
|   |-- random_forest.pkl
|   |-- svm.pkl
|-- templates/
|   |-- index.html
|-- uploads/
|-- app.py
|-- dataset.py
|-- model_training.py
|-- train_images.npy
|-- train_labels.npy
```

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/LambodarSarangi-089/Machine-Learning-for-Cat-and-Dog-Classification.git
   cd cat-dog-classification
   ```
2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask application:
   ```bash
   python app.py
   ```
4. Access the application at `http://localhost:5000/`.

   ## Requirements
- Python 3.8+
- Flask
- OpenCV
- TensorFlow/Keras
- Scikit-learn
- Matplotlib
- Pandas
- NumPy

## Notes
- The CNN model may require a GPU for faster training.
- Can Extend this project to include other pet categories or more advanced ML models.
