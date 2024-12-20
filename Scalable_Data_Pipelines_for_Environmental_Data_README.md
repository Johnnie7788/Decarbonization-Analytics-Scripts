# Scalable Data Pipelines for Environmental Data

## Introduction
The **Scalable Data Pipelines for Environmental Data** project is a robust, end-to-end ETL pipeline designed for processing large-scale environmental datasets. This pipeline leverages **Google BigQuery** for data storage and transformation and **Apache Airflow** for workflow orchestration. It empowers organizations, researchers, and policymakers to analyze, transform, and export environmental data efficiently.

---

## Key Features

- **End-to-End ETL Pipeline**:
  - Automates the extraction, transformation, and loading of environmental data.
- **Data Transformation**:
  - Aggregates metrics like average temperature, maximum CO2 emissions, and more using BigQuery SQL.
- **Scalable Architecture**:
  - Uses Google BigQuery to handle large datasets efficiently.
- **Automation**:
  - Orchestrates tasks with Apache Airflow for scheduled and monitored execution.
- **Data Export**:
  - Exports transformed data to CSV for further analysis and reporting.

---

## Repository Structure

```
Scalable-Data-Pipelines/
├── README.md                       # Project documentation (this file)
├── main_script.py                  # Python script for the pipeline
├── requirements.txt                # List of dependencies
├── data/
│   ├── raw_environmental_data.csv  # Sample raw dataset (optional)
│   ├── transformed_environmental_data.csv  # Transformed dataset (generated)
├── shap_summary.png                # SHAP summary plot (generated, optional)
└── airflow_dag.py                  # Airflow DAG definition
```

---

## Prerequisites

Ensure you have the following installed:

- Python 3.8 or higher
- Google Cloud SDK
- Apache Airflow
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up Google Cloud credentials:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
   ```

---

## How to Use

### Steps to Run the Pipeline:

1. **Prepare Your Dataset**:
   - Ensure the raw data is in CSV format and update the `RAW_DATA_PATH` constant in the script.

2. **Execute the Airflow DAG**:
   - Place the `airflow_dag.py` file in your Airflow DAGs folder.
   - Start the Airflow web server:
     ```bash
     airflow webserver
     ```
   - Start the scheduler:
     ```bash
     airflow scheduler
     ```
   - Trigger the `environmental_data_pipeline` DAG from the Airflow UI.

3. **Outputs**:
   - Transformed data is saved to `transformed_environmental_data.csv`.
   - Aggregated metrics are stored in the `transformed_environmental_data` table in BigQuery.

---

## Applications

- **Environmental Analysis**:
  - Analyze trends in climate data such as temperature, precipitation, and air quality.
- **Policy Support**:
  - Provide actionable insights for developing environmental policies.
- **Research**:
  - Support academic studies with clean, aggregated environmental datasets.

---

## Contribution

Contributions are welcome! 
---

## License

This project is licensed under the MIT License.

---

## Contact

For questions or suggestions, please reach out:
- **Email**: johnjohnsonogbidi@gmail.com
