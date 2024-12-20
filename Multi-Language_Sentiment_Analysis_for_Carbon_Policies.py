#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# Ensure reproducibility
SEED = 42
torch.manual_seed(SEED)

# Enhanced logging for better traceability
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger()

# Load Dataset
def load_dataset(file_path):
    """Loads the dataset containing texts and their corresponding labels."""
    logger.info("Loading dataset...")
    data = pd.read_csv(file_path)
    if 'text' not in data.columns or 'label' not in data.columns:
        logger.error("Dataset must contain 'text' and 'label' columns.")
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    logger.info("Dataset loaded successfully.")
    return data

# Load Model and Tokenizer
def load_model_and_tokenizer():
    """Loads the XLM-RoBERTa model and tokenizer for sentiment analysis."""
    logger.info("Loading model and tokenizer...")
    model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    logger.info("Model and tokenizer loaded successfully.")
    return tokenizer, model

# Sentiment Analysis Pipeline
def sentiment_analysis_pipeline(texts, tokenizer, model):
    """Performs sentiment analysis using the loaded model and tokenizer."""
    logger.info("Performing sentiment analysis...")
    sentiment_pipeline = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, framework="pt")
    results = sentiment_pipeline(texts, truncation=True)
    sentiments = [result['label'] for result in results]
    confidences = [result['score'] for result in results]
    logger.info("Sentiment analysis completed.")
    return sentiments, confidences

# Evaluate Model
def evaluate_model(y_true, y_pred):
    """Evaluates the model using classification metrics."""
    logger.info("Evaluating model...")
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall: {recall:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=["positive", "neutral", "negative"])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["positive", "neutral", "negative"], yticklabels=["positive", "neutral", "negative"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    logger.info("Confusion matrix saved as confusion_matrix.png.")

# Visualize Confidence Scores
def plot_confidence_distribution(confidences):
    """Plots the distribution of confidence scores."""
    logger.info("Plotting confidence score distribution...")
    plt.figure(figsize=(8, 6))
    sns.histplot(confidences, kde=True, bins=20, color="skyblue")
    plt.title("Confidence Score Distribution")
    plt.xlabel("Confidence Score")
    plt.ylabel("Frequency")
    plt.savefig("confidence_distribution.png")
    logger.info("Confidence score distribution saved as confidence_distribution.png.")

# Save Predictions
def save_predictions(data, predictions, confidences, output_path):
    """Saves the original data with predicted sentiments and confidence scores."""
    logger.info("Saving predictions...")
    data['predicted_label'] = predictions
    data['confidence'] = confidences
    data.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}.")

# Main Function
def main():
    # File paths
    dataset_path = "carbon_policies_multilang.csv"  # Replace with your dataset path
    output_path = "predicted_sentiments.csv"

    # Load dataset
    data = load_dataset(dataset_path)
    
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()

    # Perform sentiment analysis
    predictions, confidences = sentiment_analysis_pipeline(data['text'].tolist(), tokenizer, model)

    # Evaluate the model
    evaluate_model(data['label'], predictions)

    # Visualize confidence scores
    plot_confidence_distribution(confidences)

    # Save predictions
    save_predictions(data, predictions, confidences, output_path)

if __name__ == "__main__":
    main()

