# MLFLOW PROJECT
# Get conda channels
conda config --show channels

# Build a MLFlow project, if you use one entry point with name (main)
mlflow run . --experiment-name <exp-name> # here it is {readmission-prediction}

# If you have multiple entry points
mlflow run -e random_forest . --experiment-name readmission-prediction
mlflow run -e logistic_regression . --experiment-name readmission-prediction
mlflow run -e xgboost . --experiment-name readmission-prediction
mlflow run -e gradient_boosting . --experiment-name readmission-prediction
mlflow run -e decision_tree . --experiment-name readmission-prediction
mlflow run -e knn . --experiment-name readmission-prediction # Adding KNN entry point



```

```
## MLFLOW Models
``` bash
# serve the model via REST
mlflow models serve -m "path" --port 8000 --env-manager=local
mlflow models serve -m "file:///C:/Users/LAPTOP/Desktop/mlflow-main/mlflow-main/MLFlow%20Project/mlruns/551186705439064227/3a57548d8ce246e8959dd3657125d2c3/artifacts/models/XGBoost_smotetomek" --port 8001 --env-manager=local

# it will open in this link
http://localhost:8000/invocations
```

``` python
# exmaple of data to be sent


## multiple samples
{
  "dataframe_split": {
    "columns": [
      "gender",
      "age",
      "num_procedures",
      "num_medications",
      "number_emergency",
      "number_inpatient",
      "A1Cresult",
      "insulin",
      "change",
      "time_diagnoses_interaction",
      "diag_1_category",
      "diag_2_category",
      "diag_3_category",
      "sulfonylureas",
      "biguanides"
    ],
    "data": [
      [0, 80.0, 2, 0.3379, 0, 0,  2, 2, 0, 104, 2,3, 2, 1,0],
      [0, 90.0, 3, 0.21249, 0, 0, 2, 2, 0, 96, 3,0,3, 0,0],
      [0, 40.0, 2, 0.199, 0, 0, 2, 2, 1, 81, 1,3,5, 0],
      [1, 60.0, 0, 0.125, 0, 0, 2, 2, 0, 49,   0,1,0, 2,0],
      [0, 40.0, 0, 0.175, 1, 0, 2, 0, 0, 56, 3, 1,1, 0,1],
      [1, 80.0, 1, 0.375, 0, 0, 2, 2, 1,80, 3,3,3, 0,0],
      [0, 60.0, 5, 0.012499, 0, 0, 2, 2, 1, 8, 3,5,4, 0,0],
      [1, 60.0, 5, 0.15, 0, 0, 2, 3, 0, 108, 5,3,5, 0,0],
      [1, 50.0, 4, 0.199, 0, 0, 2, 2, 0, 32, 3,3, 3, 1,0]
    ]
  }
}



```

``` bash 
# if you want to use curl

curl -X POST \
  http://localhost:8000/invocations \
  -H 'Content-Type: application/json' \
  -d '{
  "dataframe_split": {
    "columns": [
      "gender",
      "age",
      "num_procedures",
      "num_medications",
      "number_emergency",
      "number_inpatient",
      "A1Cresult",
      "insulin",
      "change",
      "time_diagnoses_interaction",
      "diag_1_category",
      "diag_2_category",
      "diag_3_category",
      "sulfonylureas",
      "biguanides"
    ],
    "data": [
     [0, 80.0, 2, 0.3379, 0, 0,  2, 2, 0, 104, 2,3, 2, 1,0],
      [0, 90.0, 3, 0.21249, 0, 0, 2, 2, 0, 96, 3,0,3, 0,0],
      [0, 40.0, 2, 0.199, 0, 0, 2, 2, 1, 81, 1,3,5, 0],
      [1, 60.0, 0, 0.125, 0, 0, 2, 2, 0, 49,   0,1,0, 2,0],
      [0, 40.0, 0, 0.175, 1, 0, 2, 0, 0, 56, 3, 1,1, 0,1],
      [1, 80.0, 1, 0.375, 0, 0, 2, 2, 1,80, 3,3,3, 0,0],
      [0, 60.0, 5, 0.012499, 0, 0, 2, 2, 1, 8, 3,5,4, 0,0],
      [1, 60.0, 5, 0.15, 0, 0, 2, 3, 0, 108, 5,3,5, 0,0],
      [1, 50.0, 4, 0.199, 0, 0, 2, 2, 0, 32, 3,3, 3, 1,0]
    ]
  }




}'



# if you want to use Powershell
Invoke-RestMethod -Uri "http://localhost:8000/invocations" -Method Post -Headers @{"Content-Type" = "application/json"} -Body '{
    
  

 
  "dataframe_split": {
    "columns": [
      "gender",
      "age",
      "num_procedures",
      "num_medications",
      "number_emergency",
      "number_inpatient",
      "A1Cresult",
      "insulin",
      "change",
      "time_diagnoses_interaction",
      "diag_1_category",
      "diag_2_category",
      "diag_3_category",
      "sulfonylureas",
      "biguanides"
    ],
    "data": [0, 80.0, 2, 0.3379, 0, 0,  2, 2, 0, 104, 2,3, 2, 1,0],
      [0, 90.0, 3, 0.21249, 0, 0, 2, 2, 0, 96, 3,0,3, 0,0],
      [0, 40.0, 2, 0.199, 0, 0, 2, 2, 1, 81, 1,3,5, 0],
      [1, 60.0, 0, 0.125, 0, 0, 2, 2, 0, 49,   0,1,0, 2,0],
      [0, 40.0, 0, 0.175, 1, 0, 2, 0, 0, 56, 3, 1,1, 0,1],
      [1, 80.0, 1, 0.375, 0, 0, 2, 2, 1,80, 3,3,3, 0,0],
      [0, 60.0, 5, 0.012499, 0, 0, 2, 2, 1, 8, 3,5,4, 0,0],
      [1, 60.0, 5, 0.15, 0, 0, 2, 3, 0, 108, 5,3,5, 0,0],
      [1, 50.0, 4, 0.199, 0, 0, 2, 2, 0, 32, 3,3, 3, 1,0]
  }




}'


```

```