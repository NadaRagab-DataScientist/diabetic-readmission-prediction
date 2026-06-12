# 🏥 Diabetic Patient Readmission Prediction

An end-to-end Machine Learning project to predict diabetic patient readmission using multiple classification models with MLflow tracking, data balancing techniques, and model deployment via FastAPI (MLflow serving).

---

## 📌 Project Overview

This project aims to predict whether a diabetic patient will be readmitted based on clinical and medical features.  
It includes a full ML pipeline:

- Data preprocessing & feature engineering  
- Handling class imbalance (SMOTE / SMOTE-Tomek / Undersampling)  
- Training multiple ML models  
- Hyperparameter tuning  
- Experiment tracking using MLflow  
- Model deployment using MLflow REST API  

---

## 📊 Dataset

The dataset contains anonymized patient medical records including:

- Demographics (age, gender)
- Medical procedures
- Medications
- Lab results (A1C, insulin)
- Diagnosis categories
- Emergency & inpatient history

⚠️ The dataset is highly imbalanced (~11% readmission cases).

---

## 🧹 Data Preprocessing & Feature Engineering

- Removed irrelevant features:
  - other_combination_therapies
  - alpha_glucosidase_inhibitors
  - meglitinides
  - thiazolidinediones
  - max_glu_serum
  - binary_diabetesMed

- Applied:
  - Feature importance analysis (Decision Tree)
  - Encoding categorical variables
  - Scaling numerical features
  - Handling missing/noisy data

---

## ⚖️ Handling Class Imbalance

- Class Weighting (`class_weight="balanced"`)
- SMOTE (Synthetic Minority Oversampling)
- SMOTE-Tomek (Oversampling + Cleaning)
- Random Undersampling

---

## 🤖 Machine Learning Models

- Logistic Regression  
- Decision Tree  
- Random Forest  
- XGBoost  
- Gradient Boosting  
- KNN  
- Gaussian Naive Bayes  

---

## 📈 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC  

Focus on F1-score due to class imbalance.

---

## 🏆 Model Performance

| Model              | Accuracy |
|-------------------|----------|
| Gradient Boosting | 88.85%   |
| Random Forest     | 88.71%   |
| XGBoost           | 88.70%   |
| KNN               | 88.01%   |
| Naive Bayes       | 84.59%   |
| Decision Tree     | 80.61%   |
| Logistic Regression | 68.02% |

---

## 🔥 Final Model

**Selected Model: XGBoost**

Reasons:
- Best generalization
- Stable across resampling methods
- High F1-score for minority class
- Production-ready performance

---

## 🧪 Hyperparameter Tuning

Example (XGBoost best params):

```python
{
  'subsample': 0.8,
  'n_estimators': 200,
  'max_depth': 10,
  'learning_rate': 0.1,
  'colsample_bytree': 0.8
}
