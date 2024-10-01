import requests
import pandas as pd
import json

API_KEY = {"Authorization": "Bearer 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"}

def query(payload, api):
    API_URL = "https://api-inference.huggingface.co/models/SamLowe/roberta-base-go_emotions"
    headers = api
    json_response = requests.post(API_URL, headers=headers, json=payload)
    data = json_response.json()
    labels_scores = [(obj['label'], obj['score']) for obj in data[0]]
    df = pd.DataFrame(labels_scores, columns=['Label', 'Score'])
    return df

def go_emotions(text, api, file_path, min_sample_count = 0):
    go_emotions_result = query([text], api)
    #Load the file
    df = pd.read_csv(file_path)
    #Use the support column to filter the labels
    df = df[df['Support'] > min_sample_count]
    #Merge the dataframes
    df = df.merge(go_emotions_result, how='left', on='Label')

    return df




