import json
import boto3
import pandas as pd
import joblib
import os
import sklearn


def lambda_handler(event, context):
    model = event["queryStringParameters"]["model"]
    yil = event["queryStringParameters"]["yil"]
    km = event["queryStringParameters"]["km"]
    renk = event["queryStringParameters"]["renk"]

    model_path = "/tmp/model.pkl"
    if not os.path.exists(model_path):
        boto3.client("s3").download_file("awsbc8", "models/car.pkl", "/tmp/model.pkl")

    trained_model = joblib.load(model_path)

    input_data = pd.DataFrame([[model, yil, km, renk]])
    fiyat_prediction = trained_model.predict(input_data)

    return {
        'statusCode': 200,
        'body': json.dumps(fiyat_prediction[0])
    }
