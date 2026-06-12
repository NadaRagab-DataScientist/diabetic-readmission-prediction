import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import joblib
import base64
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_curve, auc
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTETomek
import io

# Load the models
gb_model = joblib.load(r'C:\Users\LAPTOP\Desktop\mlflow-main\mlflow-main\MLFlow Project\model_GB.pkl')  # Load pre-trained Gradient Boosting model

# Initialize Dash app
app = dash.Dash(__name__)

# Load the data
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

# Dash Layout
app.layout = html.Div([
    html.H1("Machine Learning Model Dashboard"),
    html.P("This dashboard shows the evaluation results for the selected models: Gradient Boosting and XGBoost."),
    
    dcc.Upload(
        id='upload-data',
        children=html.Button('Upload CSV'),
        multiple=False
    ),
    
    dcc.Dropdown(
        id='model-dropdown',
        options=[
            {'label': 'Gradient Boosting', 'value': 'GB'},
            {'label': 'XGBoost', 'value': 'XGB'}
        ],
        value='GB',
        style={'width': '50%'}
    ),
    
    html.Div(id='data-table'),
    
    html.Div(id='model-evaluation'),
    
    dcc.Graph(id='confusion-matrix'),
    dcc.Graph(id='roc-curve'),
    
    html.H3("Make a Prediction"),
    
    html.Div(id='input-features'),
    
    html.Button('Predict', id='predict-button'),
    html.Div(id='prediction-output')
])

@app.callback(
    Output('data-table', 'children'),
    Output('model-evaluation', 'children'),
    Output('confusion-matrix', 'figure'),
    Output('roc-curve', 'figure'),
    Output('input-features', 'children'),
    Output('prediction-output', 'children'),
    Input('upload-data', 'contents'),
    Input('model-dropdown', 'value'),
    Input('predict-button', 'n_clicks')
)
def update_dashboard(uploaded_file, model_choice, n_clicks):
    if uploaded_file is None:
        return "", "", {}, {}, "", ""
    
    # Load and clean the data
    content_type, content_string = uploaded_file.split(',')
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.BytesIO(decoded))
    
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
    
    # Resample data
    X_resampled, y_resampled = resample_data(X, y)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    
    # Choose model (GB or XGB)
    if model_choice == "GB":
        model = GradientBoostingClassifier(n_estimators=200, max_depth=10, random_state=42)
        model_name = "GradientBoosting_Model"
    else:
        model = XGBClassifier(n_estimators=200, max_depth=10, random_state=42)
        model_name = "XGBoost_Model"
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = go.Figure(data=go.Heatmap(z=cm, x=['Class 0', 'Class 1'], y=['Class 0', 'Class 1'], colorscale='Blues'))
    cm_fig.update_layout(title="Confusion Matrix", xaxis_title="Predicted Labels", yaxis_title="True Labels")
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'AUC = {roc_auc:.2f}'))
    roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line={'dash': 'dash'}))
    roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")

    # Prediction input fields
    input_features = [
        dcc.Input(id=feature, type='number', placeholder=f"Enter value for {feature}", value=0.0) for feature in X.columns
    ]
    
    prediction_output = ""
    if n_clicks:
        # Collect input data from the inputs
        input_data = [float(input_features[i].value) for i in range(len(input_features))]
        input_df = pd.DataFrame([input_data], columns=X.columns)
        prediction = model.predict(input_df)
        prediction_output = f"The predicted class for the input data is: {prediction[0]}"
    
    # Create table from first 5 rows
    table_header = [html.Th(col) for col in X.columns]
    table_rows = [html.Tr([html.Td(X.iloc[i][col]) for col in X.columns]) for i in range(5)]
    table = html.Table([html.Tr(table_header)] + table_rows)
    
    return (
        table, 
        f"Accuracy: {acc:.4f}, F1 Score: {f1:.4f}",
        cm_fig, 
        roc_fig, 
        input_features, 
        prediction_output
    )

if __name__ == '__main__':
    app.run(debug=True)
