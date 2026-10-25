# Fraud Prediction Machine Learning Pipeline

This repository contains a comprehensive Jupyter Notebook (`fraudprediction.ipynb`) designed for building, training, and evaluating machine learning pipelines for fraud detection.

## Project Overview
Fraud detection is a critical classification problem in financial and transactional systems. This project utilizes standard data science libraries and `scikit-learn` to preprocess transactional data, handle feature scaling, construct robust machine learning pipelines, and compare multiple classification algorithms.

## Dataset Source
While the notebook serves as a flexible template structure, typical open-source datasets suitable for this pipeline include:
- **Kaggle Credit Card Fraud Detection Dataset:** Features credit card transactions made by European cardholders, containing numerical input variables derived from PCA transformations.
- **IEEE-CIS Fraud Detection Dataset:** Features rich e-commerce transaction data including device information and network features.

*Note: You can plug in your dataset by loading it into a pandas DataFrame at the beginning of the notebook.*

## Libraries & Tools Included[cite: 1]
- **Data Manipulation & Analysis:** `pandas`, `numpy`
- **Data Visualization:** `matplotlib`
- **Preprocessing & Feature Engineering:** 
  - `train_test_split` for dataset partitioning
  - `MinMaxScaler` and `StandardScaler` for feature normalization and standardization
  - `ColumnTransformer` and `Pipeline` for creating streamlined, reproducible preprocessing and modeling workflows
- **Machine Learning Models:**
  - **Classification:** K-Nearest Neighbors (`KNeighborsClassifier`), Decision Tree (`DecisionTreeClassifier`), Gaussian Naive Bayes (`GaussianNB`), and Logistic Regression (`LogisticRegression`)
  - **Regression:** Linear Regression (`LinearRegression`)
- **Evaluation Metrics:** `accuracy_score`

## Requirements & Installation
Ensure you have Python installed along with the required dependencies[cite: 1]:

```bash
pip install pandas numpy matplotlib scikit-learn
