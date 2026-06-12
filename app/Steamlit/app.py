import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
from xgboost import XGBClassifier  # Import XGBoost
from sklearn.ensemble import GradientBoostingClassifier  # Import Gradient Boosting
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek

# Load the models
gb_model = joblib.load(r'C:\Users\LAPTOP\Desktop\mlflow-main\mlflow-main\MLFlow Project\model_GB.pkl')  # Load pre-trained Gradient Boosting model

# Load the data
@st.cache
def load_data(filepath):
    df = pd.read_csv(filepath)

    # Data cleaning: drop unnecessary columns
    df.drop(columns=[
        'other_combination_therapies', 
        'alpha_glucosidase_inhibitors',
        'meglitinides',
        'thiazolidinediones',
        'max_glu_serum',
        'binary_diabetesMed'
    ], inplace=True)

    # Prepare the features (X) and target (y)
    X = df.drop(columns=['readmitted'])
    y = df['readmitted']
    
    return X, y

# Resample the data (SMOTE-Tomek)
def resample_data(X, y):
    smote_tomek = SMOTETomek(random_state=42)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    return X_resampled, y_resampled

# Streamlit UI
st.title("Machine Learning Model Dashboard")
st.write("This dashboard shows the evaluation results for the selected models: Gradient Boosting and XGBoost.")

# File uploader for the user to upload their data
uploaded_file = st.file_uploader("Upload your CSV data", type=["csv"])

if uploaded_file is not None:
    # Load data
    X, y = load_data(uploaded_file)
    st.write(X.head())  # Display first 5 rows of the features
    
    # Resample data using SMOTE-Tomek
    X_resampled, y_resampled = resample_data(X, y)
    st.write("Resampled Data Shape:", X_resampled.shape)  # Display shape after resampling
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Choose model (GB or XGB)
    model_choice = st.selectbox("Choose Model", ("Gradient Boosting", "XGBoost"))

    # Train and evaluate models
    if model_choice == "Gradient Boosting":
        model = GradientBoostingClassifier(n_estimators=200, max_depth=10, random_state=42)
        model_name = "GradientBoosting_Model"
    elif model_choice == "XGBoost":
        model = XGBClassifier(n_estimators=200, max_depth=10, random_state=42)
        model_name = "XGBoost_Model"

    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    # Display results
    st.subheader(f"{model_choice} Evaluation Results")
    st.write(f"**Accuracy**: {acc:.4f}")
    st.write(f"**F1 Score**: {f1:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    st.pyplot(fig)

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
    ax_roc.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    ax_roc.plot([0, 1], [0, 1], linestyle='--')
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend(loc="lower right")
    st.pyplot(fig_roc)

    # Logging the results to MLflow
    mlflow.set_experiment("readmission-prediction")
    with mlflow.start_run(run_name=model_name):
        mlflow.log_metric('accuracy', acc)
        mlflow.log_metric('f1_score', f1)
        mlflow.sklearn.log_model(model, model_name)

    # Prediction Feature
    st.subheader("Make a Prediction")

    # Allow users to input data for prediction
    input_data = []
    for feature in X.columns:
        input_value = st.number_input(f"Enter value for {feature}", value=0.0)
        input_data.append(input_value)

    # Convert input to DataFrame
    input_df = pd.DataFrame([input_data], columns=X.columns)

    # Make prediction when button is clicked
    if st.button("Predict"):
        prediction = model.predict(input_df)
        st.write(f"The predicted class for the input data is: {prediction[0]}")
