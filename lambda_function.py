import json
import boto3
import pandas as pd
import joblib
import os
import numpy as np
from sklearn.exceptions import NotFittedError

# Constants
BUCKET_NAME = 'miull.com'
LOCAL_MODEL_PATH = "/tmp/car.pkl"
REMOTE_MODEL_PATH = "car.pkl"
ALLOWED_ORIGINS = ['https://miuulapp.github.rocks', 'http://localhost:3011']

def lambda_handler(event, context):
    # CORS handling
    origin = event['headers'].get('origin', '')

    # Validating and extracting query parameters
    query_params = event.get("queryStringParameters", {})
    required_params = ["model", "yil", "km", "renk"]
    if not all(param in query_params for param in required_params):
        return {"statusCode": 400, "body": "Missing required query parameters"}

    try:
        model, yil, km, renk = [query_params[param] for param in required_params]
        yil, km = int(yil), int(km)  # Ensure yil and km are integers
    except ValueError:
        return {"statusCode": 400, "body": "Invalid parameter types"}

    # Load model
    if not os.path.exists(LOCAL_MODEL_PATH):
        boto3.client("s3").download_file(BUCKET_NAME, REMOTE_MODEL_PATH, LOCAL_MODEL_PATH)

    try:
        trained_model = joblib.load(LOCAL_MODEL_PATH)
    except (OSError, joblib.externals.loky.process_executor.TerminatedWorkerError):
        return {"statusCode": 500, "body": "Error loading model"}

    # Predict
    try:
        input_data = pd.DataFrame([[model, yil, km, renk]])
        fiyat_prediction = trained_model.predict(input_data)
        fiyat = int(np.round(fiyat_prediction[0]))
    except NotFittedError:
        return {"statusCode": 500, "body": "Model not fitted"}
    except Exception as e:
        return {"statusCode": 500, "body": f"Prediction error: {str(e)}"}

    # Response
    return {
        "statusCode": 200,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": origin
        },
        "body": json.dumps({"fiyat": fiyat})
    }

