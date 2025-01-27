import json
import boto3
from transformers import AutoTokenizer
from io import BytesIO
import os

s3_client = boto3.client('s3')
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_data(examples):
    input_texts = [f"question: {q}" for q in examples["question"]]
    target_texts = examples["answer"]

    model_inputs = tokenizer(
        input_texts, 
        text_target=target_texts,
        truncation=True, 
        padding='max_length',
        max_length=50,
        return_tensors="pt"
    )

    return model_inputs

def lambda_handler(event, context):
    bucket_name = event['Records'][0]['s3']['bucket']['name']
    object_key = event['Records'][0]['s3']['object']['key']

    file_obj = s3_client.get_object(Bucket=bucket_name, Key=object_key)
    file_content = file_obj['Body'].read().decode('utf-8')

    examples = json.loads(file_content)
    
    tokenized_data = preprocess_data(examples)

    output_bucket = os.environ['PROCESSED_BUCKET_NAME']
    output_key = f"tokenized/{object_key}"

    tokenized_json = json.dumps(tokenized_data, default=str)
    s3_client.put_object(Bucket=output_bucket, Key=output_key, Body=tokenized_json)

    return {
        'statusCode': 200,
        'body': json.dumps('Tokenization completed successfully.')
    }
