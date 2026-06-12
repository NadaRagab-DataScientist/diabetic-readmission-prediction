from flask import Flask, request, jsonify
import mlflow
import mlflow.sklearn
import pandas as pd
import datetime
import json

app = Flask(__name__)

# تحميل الموديل
model_uri = "runs:/b0046f6bdd92495d8bbbaa3b31c12750/models/GradientBoosting_smotetomek"  # غير <your_run_id>
model = mlflow.sklearn.load_model(model_uri)

@app.route('/predict', methods=['POST'])
def predict():
    # تحقق إذا فيه ملف مرفوع
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    # قراءة محتوى الملف
    file_content = file.read().decode('utf-8')

    # نحول النص لـ JSON
    try:
        input_json = json.loads(file_content)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON in file"}), 400

    # قراءة الداتا وتحويلها لـ DataFrame
    if "dataframe_split" not in input_json:
        return jsonify({"error": "Invalid input format"}), 400

    columns = input_json["dataframe_split"]["columns"]
    data = input_json["dataframe_split"]["data"]
    input_df = pd.DataFrame(data, columns=columns)

    # عمل prediction
    predictions = model.predict(input_df)

    # تسجيل الـ input والـ output في MLflow
    with mlflow.start_run(run_name="Inference_logging", nested=True):
        mlflow.log_dict(input_json, "input_data.json")
        for idx, pred in enumerate(predictions):
            mlflow.log_metric(f"prediction_{idx}", pred)
        mlflow.log_param("timestamp", str(datetime.datetime.now()))
    
    return jsonify({"predictions": predictions.tolist()})

if __name__ == '__main__':
    app.run(port=5000)
