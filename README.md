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
|-- datasets/  (Placeholders for downloaded datasets)
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
|-- README.md
```

1. Data Collection
   - Download the labeled dataset of cat and dog images from Kaggle:
     [Dog vs Cat Dataset on Kaggle](https://www.kaggle.com/datasets/anthonytherrien/dog-vs-cat)
   - Extract and place the downloaded dataset into the datasets/ directory.

2. Preprocessing
   - Resize images to a uniform size (e.g., 128x128 pixels).
   - Flatten images into 1D arrays for traditional ML models (SVM, Random Forest, etc.).
   - Normalize pixel values to a range of 0-1.
   - Save preprocessed data (e.g., train_images.npy and train_labels.npy) for reuse.

3. Model Training
   - Train the following models:
     - Support Vector Machine (SVM): Use sklearn's SVC.
     - Random Forest: Use sklearn's RandomForestClassifier.
     - Logistic Regression: Use sklearn's LogisticRegression.
     - Convolutional Neural Network (CNN): Build with TensorFlow/Keras.
     - K-means Clustering: For unsupervised classification.
   - Save the trained models in the saved_models/ directory.

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/LambodarSarangi-089/Machine-Learning-for-Cat-and-Dog-Classification.git
   cd Machine-Learning-for-Cat-and-Dog-Classification
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
