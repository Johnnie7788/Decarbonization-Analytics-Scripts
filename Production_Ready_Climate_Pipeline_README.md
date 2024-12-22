
# Production Ready Climate Pipeline

## Overview
The **Production Ready Climate Pipeline** is a scalable pipeline designed to process climate-related data and generate actionable insights. The system integrates state-of-the-art machine learning techniques, real-time monitoring, and deployment practices to support decarbonization efforts.

## Features
- **Data Validation:** Schema validation using `pydantic` to ensure data integrity.
- **Scalable Data Processing:** Supports large datasets with Dask for efficient computations.
- **Machine Learning:** Utilizes `RandomizedSearchCV` for hyperparameter optimization.
- **Model Versioning:** Tracks model parameters and metrics with MLflow.
- **Secure Secrets Management:** Protects sensitive credentials with `.env` files.
- **Real-Time Monitoring:** Tracks metrics using Prometheus.
- **Testing:** Includes unit tests with `pytest` and AWS S3 mocking using `moto`.
- **Deployment Ready:** Modular design compatible with CI/CD pipelines and containerization.

## Installation
### Prerequisites
- Python 3.8+
- Pip or Conda for package management
- Docker (optional, for containerization)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set Up Environment Variables
Create a `.env` file in the root directory and populate it with the following:
```env
AWS_ACCESS_KEY=<your_aws_access_key>
AWS_SECRET_KEY=<your_aws_secret_key>
AWS_BUCKET=<your_aws_bucket_name>
DATABASE_URI=<your_database_uri>
```

### Configuration
Modify the `config.yaml` file to specify the following:
```yaml
data_path: "path/to/your/data.csv"
output_path: "path/to/output"
database_uri: "database_uri_placeholder"
model_path: "path/to/save/model.pkl"
aws_bucket: "aws_bucket_name_placeholder"
```

## Usage
### Run the Pipeline
```bash
python main.py
```

### Monitor Metrics
Start the Prometheus server:
```bash
prometheus --config.file=prometheus.yml
```
Access metrics at `http://localhost:8000`.

### Run Tests
```bash
pytest
```

## File Structure
```
.
├── config.yaml           # Configuration file
├── main.py               # Main pipeline script
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
├── .env                  # Environment variables
├── tests/                # Unit tests
├── docker/               # Dockerfile (optional)
```

## Key Components
### Data Validation
- Ensures data integrity with schema-based checks using `pydantic`.

### Machine Learning
- Trains and tunes models with `RandomizedSearchCV`.
- Tracks performance and versions with MLflow.

### Monitoring
- Real-time metrics tracked using Prometheus.

### Deployment
- Supports deployment to AWS with S3 integration and Docker compatibility.

## Contributing
Contributions are welcome! 

## License
This project is licensed under the MIT License. 

## Contact
For questions or suggestions, please contact:
- **Email:** johnjohnsonogbidi@gmail.com

