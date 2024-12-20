#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.cloud import bigquery
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryExecuteQueryOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
import pandas as pd
import os

# Google Cloud Credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/credentials.json"

# Constants
PROJECT_ID = "your-project-id"
DATASET_ID = "your-dataset-id"
TABLE_ID = "environmental_data"
RAW_DATA_PATH = "data/raw_environmental_data.csv"

# Function to Load Data to BigQuery
def load_data_to_bigquery():
    """Loads raw environmental data to Google BigQuery."""
    print("Loading data to BigQuery...")
    client = bigquery.Client()

    table_ref = client.dataset(DATASET_ID).table(TABLE_ID)
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        autodetect=True,
    )

    with open(RAW_DATA_PATH, "rb") as source_file:
        load_job = client.load_table_from_file(source_file, table_ref, job_config=job_config)

    load_job.result()  # Wait for the job to complete
    print(f"Loaded {load_job.output_rows} rows to {TABLE_ID}.")

# SQL Query for Transformation
TRANSFORM_QUERY = f"""
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.transformed_environmental_data` AS
SELECT
    region,
    AVG(temperature) AS avg_temperature,
    AVG(precipitation) AS avg_precipitation,
    MAX(co2_emissions) AS max_co2_emissions,
    MIN(air_quality_index) AS min_air_quality_index
FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`
GROUP BY region;
"""

# Define the Airflow DAG
default_args = {
    'owner': 'airflow',
    'retries': 2,
}

dag = DAG(
    "environmental_data_pipeline",
    default_args=default_args,
    description="A scalable ETL pipeline for processing environmental data.",
    schedule_interval="@daily",
    start_date=days_ago(1),
    catchup=False,
)

# Task 1: Load Data to BigQuery
load_data_task = PythonOperator(
    task_id="load_data_to_bigquery",
    python_callable=load_data_to_bigquery,
    dag=dag,
)

# Task 2: Transform Data in BigQuery
transform_data_task = BigQueryExecuteQueryOperator(
    task_id="transform_data_in_bigquery",
    sql=TRANSFORM_QUERY,
    use_legacy_sql=False,
    dag=dag,
)

# Task 3: Export Transformed Data to CSV
def export_transformed_data():
    """Exports transformed data from BigQuery to a CSV file."""
    print("Exporting transformed data from BigQuery...")
    client = bigquery.Client()
    query = f"SELECT * FROM `{PROJECT_ID}.{DATASET_ID}.transformed_environmental_data`"

    query_job = client.query(query)
    results = query_job.result()

    rows = [dict(row) for row in results]
    df = pd.DataFrame(rows)

    export_path = "data/transformed_environmental_data.csv"
    df.to_csv(export_path, index=False)
    print(f"Transformed data exported to {export_path}.")

export_data_task = PythonOperator(
    task_id="export_transformed_data",
    python_callable=export_transformed_data,
    dag=dag,
)

# Task Dependencies
load_data_task >> transform_data_task >> export_data_task

