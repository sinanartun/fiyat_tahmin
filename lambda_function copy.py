import json
import boto3
import pandas as pd
import joblib
import os
import numpy as np
import sklearn

bucket_name = 'busonbucket'
local_model_path = "/tmp/car.pkl"
remote_model_path = "model_car.pkl"


def lambda_handler(event, context):
    origin = event['headers'].get('origin', '')
    allowed_origins = ['https://miuulapp.github.rocks', 'http://localhost:3011']
    access_control_allow_origin = ''
    if origin in allowed_origins:
        access_control_allow_origin = origin
    model = event["queryStringParameters"]["model"]
    yil = event["queryStringParameters"]["yil"]
    km = event["queryStringParameters"]["km"]
    renk = event["queryStringParameters"]["renk"]

    if not os.path.exists(local_model_path):
        boto3.client("s3").download_file(bucket_name, remote_model_path, local_model_path)

    trained_model = joblib.load(local_model_path)

    input_data = pd.DataFrame([[model, yil, km, renk]])
    fiyat_prediction = trained_model.predict(input_data)

    # Convert numpy.float64 or numpy.ndarray to int
    if type(fiyat_prediction[0]) == np.ndarray and len(fiyat_prediction[0]) == 1:
        fiyat = int(fiyat_prediction[0].item())
    else:
        fiyat = int(fiyat_prediction[0])

   
    return {
            "statusCode": 200,
            "headers": {"content-type": "application/json", "Access-Control-Allow-Origin": access_control_allow_origin},
            "body": json.dumps(fiyat, indent=0, sort_keys=True, default=str)
        }

