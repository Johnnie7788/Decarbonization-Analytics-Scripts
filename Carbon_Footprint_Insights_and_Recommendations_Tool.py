#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, silhouette_score
import plotly.express as px

# ==========================
# Carbon Footprint Insights and Recommendations Tool
# ==========================

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load and preprocess the data
def load_data(file_path):
    """Load dataset, handle missing values, and validate data."""
    logging.info("Loading data from file.")
    df = pd.read_csv(file_path)
    if df.isnull().values.any():
        logging.warning("Missing values detected. Filling with mean values.")
        df.fillna(df.mean(), inplace=True)
    if df.empty:
        raise ValueError("The input dataset is empty. Please provide a valid dataset.")
    logging.info("Data successfully loaded and validated.")
    return df

# Calculate carbon footprint
def calculate_carbon_footprint(df):
    """Calculate total emissions based on activity data and emission factors."""
    logging.info("Calculating carbon footprint.")
    if 'Emission_Factor' not in df.columns or 'Activity_Data' not in df.columns:
        raise KeyError("Required columns 'Emission_Factor' or 'Activity_Data' are missing.")
    df['Total_Emissions'] = df['Emission_Factor'] * df['Activity_Data']
    logging.info("Carbon footprint calculated.")
    return df

# Train a prediction model with hyperparameter tuning
def train_prediction_model(df):
    """Train a Random Forest model with optimized hyperparameters."""
    logging.info("Training the prediction model.")
    features = ['Activity_Data', 'Emission_Factor']
    target = 'Total_Emissions'
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameter tuning using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    logging.info(f"Optimized Model Mean Squared Error: {mse:.2f}")
    logging.info(f"Best Model Parameters: {grid_search.best_params_}")
    return best_model

# Cluster data for insights
def cluster_data(df):
    """Perform clustering to identify high-emission activities."""
    logging.info("Performing clustering on data.")
    scaler = StandardScaler()
    features = df[['Total_Emissions', 'Activity_Data']]
    scaled_features = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Cluster'] = kmeans.fit_predict(scaled_features)

    silhouette_avg = silhouette_score(scaled_features, df['Cluster'])
    logging.info(f"Silhouette Score: {silhouette_avg:.2f}")
    logging.info("Clustering complete.")
    return df, kmeans

# Generate automatic recommendations
def generate_recommendations(df):
    """Provide actionable recommendations based on clusters."""
    logging.info("Generating recommendations based on clusters.")
    recommendations = []
    for _, row in df.iterrows():
        if row['Cluster'] == 2:  # Assuming cluster 2 is high emissions
            recommendations.append(f"Reduce emissions by optimizing activity: {row['Activity']}.")
        elif row['Cluster'] == 1:
            recommendations.append(f"Monitor activity: {row['Activity']}, potential for optimization.")
        else:
            recommendations.append(f"Maintain activity: {row['Activity']} with current emission levels.")
    df['Recommendations'] = recommendations
    logging.info("Recommendations generated.")
    return df

# Visualize insights with Plotly
def visualize_insights(df):
    """Create interactive visualizations for emission data and clusters."""
    logging.info("Creating interactive visualizations.")
    fig = px.scatter(
        df, 
        x='Activity_Data', 
        y='Total_Emissions', 
        color='Cluster', 
        size='Total_Emissions', 
        hover_data=['Recommendations'],
        title='Cluster Analysis of Emissions'
    )
    fig.update_layout(
        xaxis_title='Activity Data',
        yaxis_title='Total Emissions',
        legend_title='Cluster'
    )
    fig.show()
    logging.info("Visualization created.")

# Save results
def save_results(df, output_path):
    """Save the processed data and recommendations to a file."""
    df.to_csv(output_path, index=False)
    logging.info(f"Results saved to {output_path}")

# Main pipeline
def main():
    """Main function to execute the carbon footprint tool."""
    input_file = 'emission_data.csv'  # Replace with your dataset
    output_file = 'carbon_footprint_insights.csv'

    try:
        # Step 1: Load data
        data = load_data(input_file)

        # Step 2: Calculate carbon footprint
        data = calculate_carbon_footprint(data)

        # Step 3: Train prediction model
        model = train_prediction_model(data)

        # Step 4: Cluster data
        data, kmeans_model = cluster_data(data)

        # Step 5: Generate recommendations
        data = generate_recommendations(data)

        # Step 6: Visualize insights
        visualize_insights(data)

        # Step 7: Save results
        save_results(data, output_file)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

