#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import logging
import yaml
from sqlalchemy import create_engine
from dask import dataframe as dd
from pathlib import Path
import boto3  # AWS SDK for S3 integration
import joblib
from pydantic import BaseModel, ValidationError
import pytest
import mlflow
from dotenv import load_dotenv
from prometheus_client import start_http_server, Summary, Counter
import docker  # Optional: For Docker integration
from moto import mock_s3  # Mock AWS S3 for testing

# Load environment variables for secure credential management
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load configuration from a YAML file
CONFIG_FILE = "config.yaml"

if not os.path.exists(CONFIG_FILE):
    logging.error("Configuration file is missing. Please provide a config.yaml file.")
    raise FileNotFoundError("Configuration file not found.")

with open(CONFIG_FILE, 'r') as file:
    config = yaml.safe_load(file)

# Constants
DATA_PATH = config['data_path']
OUTPUT_PATH = config['output_path']
DATABASE_URI = os.getenv("DATABASE_URI", config['database_uri'])
MODEL_PATH = config['model_path']
AWS_BUCKET = os.getenv("AWS_BUCKET", config['aws_bucket'])
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

# Prometheus metrics
REQUEST_TIME = Summary('request_processing_seconds', 'Time spent processing request')
FAILURE_COUNT = Counter('pipeline_failures', 'Number of pipeline failures')

# Initialize AWS session
def get_s3_client():
    if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
        logging.error("AWS credentials are missing. Ensure they are set in the environment variables.")
        raise EnvironmentError("Missing AWS credentials.")
    return boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# Data schema using Pydantic
class EmissionDataSchema(BaseModel):
    emission_source: str
    emission_value: float
    policy_impact: str

# Utility: Data validation function
def validate_data(df, schema):
    logging.info("Validating data...")
    for index, record in df.iterrows():
        try:
            schema(**record.to_dict())
        except ValidationError as e:
            logging.error(f"Validation error at row {index}: {e}")
            FAILURE_COUNT.inc()
            raise

# Utility: Save data to a database
def save_to_database(df, table_name, database_uri):
    logging.info(f"Saving data to table {table_name} in database...")
    engine = create_engine(database_uri)
    with engine.begin() as connection:
        df.to_sql(table_name, con=connection, if_exists='replace', index=False)

# Utility: Upload files to AWS S3
def upload_to_s3(file_path, bucket_name, key):
    logging.info(f"Uploading {file_path} to AWS S3 bucket {bucket_name}...")
    s3_client = get_s3_client()
    try:
        s3_client.upload_file(file_path, bucket_name, key)
    except Exception as e:
        logging.error(f"Failed to upload {file_path} to S3: {e}")
        FAILURE_COUNT.inc()
        raise

# Step 1: Load and preprocess data
@REQUEST_TIME.time()
def load_and_preprocess(file_path):
    logging.info("Loading and preprocessing data...")
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.parquet'):
        df = pd.read_parquet(file_path)
    else:
        raise ValueError("Unsupported file format. Please use CSV or Parquet.")

    validate_data(df, EmissionDataSchema)

    df['emission_value'] = pd.to_numeric(df['emission_value'], errors='coerce')
    df.dropna(inplace=True)

    return df

# Step 2: Build and train ML model
@REQUEST_TIME.time()
def train_model(df):
    logging.info("Training machine learning model...")
    X = df.drop(columns=['emission_value'])
    y = df['emission_value']

    X = pd.get_dummies(X, drop_first=True)  # One-hot encoding for categorical data

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    randomized_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=10, cv=5, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42)
    randomized_search.fit(X_train, y_train)

    best_model = randomized_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Model training complete. MSE: {mse:.2f}")

    # Save the trained model using MLflow
    with mlflow.start_run():
        mlflow.log_param("best_model", str(best_model))
        mlflow.log_metric("mse", mse)
        mlflow.sklearn.log_model(best_model, "model")

    joblib.dump(best_model, MODEL_PATH)
    logging.info(f"Model saved to {MODEL_PATH}")

    return best_model

# Step 3: Deploy the pipeline
@REQUEST_TIME.time()
def deploy_pipeline():
    try:
        # Load data
        raw_data_path = Path(DATA_PATH)
        if not raw_data_path.exists():
            logging.error(f"Data path {raw_data_path} does not exist.")
            FAILURE_COUNT.inc()
            raise FileNotFoundError(f"Data path {raw_data_path} does not exist.")

        df = load_and_preprocess(raw_data_path)

        # Train and save the model
        trained_model = train_model(df)

        # Save processed data to a database
        save_to_database(df, "processed_emissions", DATABASE_URI)

        # Upload the model to AWS S3
        upload_to_s3(MODEL_PATH, AWS_BUCKET, f"models/{Path(MODEL_PATH).name}")

        logging.info("Pipeline deployment complete.")
    except Exception as e:
        logging.error(f"Pipeline deployment failed: {e}")
        FAILURE_COUNT.inc()
        raise

# Main execution
def main():
    start_http_server(8000)  # Start Prometheus monitoring server
    logging.info("Starting Production Ready Climate Pipeline...")
    deploy_pipeline()
    logging.info("Pipeline executed successfully.")

if __name__ == "__main__":
    main()

# Test case example using pytest
@mock_s3
def test_upload_to_s3():
    s3_client = get_s3_client()
    s3_client.create_bucket(Bucket=AWS_BUCKET)
    try:
        upload_to_s3("test_file.txt", AWS_BUCKET, "test/test_file.txt")
        assert True
    except Exception:
        assert False

