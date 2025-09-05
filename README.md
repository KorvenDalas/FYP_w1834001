# Bankruptcy Prediction with Artificial Neural Networks

This repository contains my BSc (Hons) Data Science and Analytics final project: developing and evaluating Artificial Neural Networks (ANNs) for bankruptcy prediction using financial statement data.

This project highlights skills in data preprocessing, machine learning, and financial analytics. I tackle real-world challenges such as dealing with imbalanced datasets, handling missing data, and ensuring that the models are interpretable.

## Project Overview
Goal: Predict the probability of bankruptcy for companies using historical financial indicators.  
Dataset: Polish Companies Bankruptcy Data (UCI ML Repository)  
Techniques:
 - Data imputation (Mean, KNN, Linear Regression)
 - Class balancing with SMOTE
 - ANN architectures (simple → deep, with BatchNorm & Dropout)
 - Benchmark models: Random Forest and XGBoost

## 📊 Results
Built and compared 6 ANN architectures across multiple imputation strategies.  
Tackled a highly imbalanced dataset (95% solvent companies vs. 5% bankrupt).  
Best-performing models:
 - ANN (64-1024-1024-32-32-1) – Accuracy ~94%
 - XGBoost – Accuracy ~97.6%, Precision ~81%
Key insight: traditional ML methods (XGBoost, Random Forest) can outperform deep models in financial prediction tasks, depending on data characteristics.

## 🛠️ Tools & Libraries
Python (Google Colab).  
Libraries: Pandas, NumPy, Scikit-learn, Keras, Imbalanced-learn (SMOTE), Missingno, SciPy.  
ML techniques: ANN, Random Forest, XGBoost, SMOTE oversampling, feature scaling, imputation.

## 🚀 Skills Demonstrated
Data preprocessing & handling 53% missing values in the dataset  
Balancing imbalanced datasets with advanced sampling techniques  
Neural network design, hyperparameter tuning, and training  
Model evaluation with accuracy, precision, recall, and trade-off analysis  
Applying data science to financial risk & fintech use cases  
