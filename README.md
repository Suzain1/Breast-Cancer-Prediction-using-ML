🩺 Breast Cancer Prediction Model

This repository contains a Python script for predicting breast cancer using various machine learning algorithms. The code is designed to load a dataset, preprocess the data, train multiple models, and evaluate their performance.

📁 File Structure

Breast Cancer Prediction.py: The main script that includes all the steps for data loading, preprocessing, model training, and evaluation.

📊 Data Preprocessing

The script performs the following data preprocessing steps:

Data Loading: Reads the breast cancer dataset from a CSV file.

Data Splitting: Splits the dataset into training and testing sets.

Feature Scaling: Standardizes the features using StandardScaler.

🤖 Machine Learning Models

The script trains and evaluates the following machine learning models:

Logistic Regression

Naive Bayes (GaussianNB)

Support Vector Machine (SVM)

Each model is trained on the training data, and its performance is evaluated on the test data using accuracy, classification report, and confusion matrix.

📊 Model Evaluation

The script generates performance metrics and visualizations to evaluate each model, including:

Accuracy Score: A measure of the correctness of the predictions.

Classification Report: Includes precision, recall, and F1-score for each class.

Confusion Matrix: A matrix showing the true vs predicted classifications.
Visualizations such as confusion matrices are also plotted using Seaborn and Matplotlib.

