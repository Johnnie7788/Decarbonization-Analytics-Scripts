#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from transformers import pipeline as hf_pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import logging

# Setup logging for debugging and transparency
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Utility function for loading data
def load_data(file_path, required_columns=None):
    try:
        data = pd.read_csv(file_path)
        logging.info(f"Data loaded successfully from {file_path}.")
        if required_columns:
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
        return data
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

# Step 1: Emission Reduction Prediction
def emission_reduction_prediction(data_path):
    logging.info("Starting Emission Reduction Prediction Module.")
    data = load_data(data_path, required_columns=["emission_reduction"])

    # Data preprocessing
    if data.isnull().sum().sum() > 0:
        logging.warning("Missing values detected. Filling with median values.")
        data.fillna(data.median(), inplace=True)

    X = data.drop(columns=["emission_reduction"])
    y = data["emission_reduction"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

    logging.info("Performing hyperparameter tuning for Random Forest...")
    grid_search.fit(X_train, y_train)

    best_rf_model = grid_search.best_estimator_
    y_pred = best_rf_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Optimized MSE for Emission Reduction Prediction: {mse:.2f}")

    # Feature importance visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(x=best_rf_model.feature_importances_, y=X.columns)
    plt.title("Feature Importances")
    plt.show()

# Step 2: Policy Sentiment Analysis
def policy_sentiment_analysis(data_path):
    logging.info("Starting Policy Sentiment Analysis Module.")
    data = load_data(data_path, required_columns=["policy_text", "sentiment"])

    # Data validation
    if data.isnull().sum().sum() > 0:
        logging.warning("Missing values detected in policy data. Dropping rows.")
        data.dropna(inplace=True)

    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X_text = vectorizer.fit_transform(data["policy_text"])
    y_text = data["sentiment"]

    X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text, y_text, test_size=0.2, random_state=42)

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_text, y_train_text)
    y_pred_text = logistic_model.predict(X_test_text)

    report = classification_report(y_test_text, y_pred_text)
    logging.info(f"Sentiment Analysis Classification Report:\n{report}")

    # Transformer-based sentiment analysis
    transformer_sentiment = hf_pipeline("sentiment-analysis")
    example_policy = "This policy aims to significantly reduce carbon emissions by promoting renewable energy."
    sentiment_result = transformer_sentiment(example_policy)
    logging.info(f"Transformer Sentiment Analysis Result: {sentiment_result}")

# Step 3: Emission Hotspot Identification
def emission_hotspot_identification():
    logging.info("Starting Emission Hotspot Identification Module.")
    latitude = np.random.uniform(-90, 90, 1000)
    longitude = np.random.uniform(-180, 180, 1000)
    emissions = np.random.uniform(100, 10000, 1000)

    geo_data = pd.DataFrame({"latitude": latitude, "longitude": longitude, "emissions": emissions})

    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=geo_data, x="longitude", y="latitude", size="emissions", hue="emissions", palette="Reds", legend=False)
    plt.title("Emission Hotspots")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

# Step 4: Main Execution
def main():
    # Define paths to datasets
    emission_data_path = "carbon_emission_data.csv"
    policy_data_path = "policy_texts.csv"

    # Run all modules
    emission_reduction_prediction(emission_data_path)
    policy_sentiment_analysis(policy_data_path)
    emission_hotspot_identification()

if __name__ == "__main__":
    main()

