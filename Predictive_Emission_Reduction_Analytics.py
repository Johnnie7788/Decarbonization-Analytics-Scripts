#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import joblib
import shap
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Ensure reproducibility
SEED = 42
np.random.seed(SEED)

# Load Dataset
def load_data(file_path):
    """Loads the dataset and performs basic validation."""
    logger.info("Loading dataset...")
    data = pd.read_csv(file_path)
    if 'emission_reduction' not in data.columns:
        logger.error("Dataset must contain 'emission_reduction' column.")
        raise ValueError("Dataset must contain 'emission_reduction' column.")
    logger.info("Dataset loaded successfully.")
    return data

# Validate Dataset
def validate_data(data):
    """Validates the dataset for missing values and outliers."""
    if data.isnull().sum().any():
        logger.error("Dataset contains missing values. Please handle them before proceeding.")
        raise ValueError("Dataset contains missing values.")
    if data.select_dtypes(include=[np.number]).apply(lambda x: (x > x.quantile(0.99)).sum()).sum() > 0:
        logger.warning("Dataset contains potential outliers.")
    logger.info("Dataset validation completed.")

# Preprocess Data
def preprocess_data(data):
    """Prepares data for training and testing."""
    logger.info("Preprocessing data...")
    X = data.drop(columns=['emission_reduction'])
    y = data['emission_reduction']
    logger.info("Data preprocessing completed.")
    return X, y

# Train Model
def train_model(X_train, y_train):
    """Trains a Random Forest Regressor with hyperparameter tuning."""
    logger.info("Training model...")
    rf = RandomForestRegressor(random_state=SEED)

    # Hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5, 10]
    }

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='r2', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info("Model training completed.")
    return grid_search.best_estimator_

# Evaluate Model
def evaluate_model(model, X_test, y_test):
    """Evaluates the model on the test set and returns metrics."""
    logger.info("Evaluating model...")
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    logger.info(f"Mean Squared Error: {mse:.4f}")
    logger.info(f"Mean Absolute Error: {mae:.4f}")
    logger.info(f"R-squared: {r2:.4f}")

    return predictions, mse, mae, r2

# SHAP Analysis
def explain_model(model, X_train, feature_names):
    """Performs SHAP analysis to explain model predictions."""
    logger.info("Generating SHAP explanations...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    shap.summary_plot(shap_values, X_train, feature_names=feature_names, plot_type="bar", show=False)
    plt.savefig("shap_summary.png", bbox_inches="tight")
    logger.info("SHAP summary plot saved to shap_summary.png.")

# Plot Results
def plot_results(y_test, predictions):
    """Plots actual vs predicted emission reductions."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions, alpha=0.7, edgecolors="k")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel("Actual Emission Reduction")
    plt.ylabel("Predicted Emission Reduction")
    plt.title("Actual vs Predicted Emission Reductions")
    plt.savefig("actual_vs_predicted.png", bbox_inches="tight")
    logger.info("Actual vs Predicted plot saved to actual_vs_predicted.png.")

# Feature Importance
def plot_feature_importance(model, feature_names):
    """Plots the feature importance of the Random Forest model."""
    logger.info("Generating feature importance plot...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 8))
    plt.title("Feature Importance")
    plt.bar(range(len(feature_names)), importances[indices], align="center")
    plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=45)
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.savefig("feature_importance.png", bbox_inches="tight")
    logger.info("Feature importance plot saved to feature_importance.png.")

# Save Model
def save_model(model, file_path):
    """Saves the trained model to a file."""
    logger.info("Saving model...")
    joblib.dump(model, file_path)
    logger.info(f"Model saved to {file_path}.")

# Main Function
def main():
    # File paths
    dataset_path = "emission_reduction_data.csv"  # Replace with your dataset path
    model_path = "emission_reduction_model.pkl"

    # Load and validate data
    data = load_data(dataset_path)
    validate_data(data)

    # Preprocess data
    X, y = preprocess_data(data)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    predictions, mse, mae, r2 = evaluate_model(model, X_test, y_test)

    # Plot results
    plot_results(y_test, predictions)

    # SHAP analysis
    explain_model(model, X_train, feature_names=X.columns)

    # Feature importance
    plot_feature_importance(model, feature_names=X.columns)

    # Save model
    save_model(model, model_path)

if __name__ == "__main__":
    main()

