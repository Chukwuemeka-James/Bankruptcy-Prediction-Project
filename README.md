# Bankruptcy Prediction Using Various Machine Learning Models

## Project Overview
This project focuses on predicting the likelihood of bankruptcy for clients based on financial data. The primary goal is to leverage different classification models to assess whether a given client is at risk of bankruptcy. By experimenting with multiple machine learning models and applying dimensionality reduction techniques, the project evaluates how different models perform before and after reducing the data's dimensionality using Principal Component Analysis (PCA).

## Dataset Description
The dataset consists of financial client data, including various financial metrics that are used to predict bankruptcy. The target variable `Bankrupt?` indicates whether a client is bankrupt (`1`) or not (`0`).

### Dataset Features:
- **Features**: A variety of financial indicators (e.g., total assets, liabilities, net worth).
- **Target**: 
  - `0` indicates the client is not bankrupt.
  - `1` indicates the client is bankrupt.

### Preprocessing Steps:
1. **Drop Irrelevant Features**: The `Net Income Flag` column is dropped since it contains only a single value.
2. **Label Encoding**: The target variable `Bankrupt?` is converted into categorical labels (`YES` for bankruptcy and `NO` for non-bankruptcy).
3. **Train-Test Split**: The dataset is split into training and test sets, with 70% of the data used for training and 30% for testing.
4. **Feature Scaling**: The features are scaled using `StandardScaler` to ensure all values are normalized for model training.

## Project Motivation
Financial institutions need to assess the risk of bankruptcy for clients to manage credit risk. This project aims to predict bankruptcy using machine learning models, evaluate their performance, and explore how dimensionality reduction techniques such as PCA affect model accuracy. Understanding the impact of dimensionality reduction is important when dealing with large datasets, where some features may be redundant.

## Methodology

### 1. **Data Preprocessing**
### 2. **Model Training**
A variety of classification models were trained using the processed dataset:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree
- Support Vector Machines (SVM) – Linear and RBF kernels
- Neural Network
- Random Forest
- Gradient Boosting

### 3. **Dimensionality Reduction with PCA**
PCA is applied to reduce the dataset's dimensionality, ensuring that only the most important features are used for training.

### 4. **Performance Comparison Before and After PCA**
The change in model performance is visualized to observe how dimensionality reduction affected accuracy.


## Results
- **Model Performance**: 
  - The initial models, before PCA, provided varying levels of accuracy, with models like Random Forest and Gradient Boosting performing better.
  - After applying PCA, the performance of some models improved, while others showed a slight drop in accuracy due to the reduced dimensionality.

- **Dimensionality Reduction**: 
  - PCA helped to reduce the number of features while retaining a significant portion of the variance, thereby simplifying the models.

## Tools and Technologies
- Python
- Scikit-learn (Logistic Regression, KNN, Decision Tree, SVM, Neural Network, Random Forest, Gradient Boosting)
- PCA for Dimensionality Reduction
- Plotly for Visualization

## Future Work
Future improvements can include:
- Hyperparameter tuning of models to further optimize performance.
- Exploring more sophisticated dimensionality reduction techniques.
- Addressing the class imbalance in the dataset by implementing resampling techniques or using specialized algorithms.
