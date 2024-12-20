#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import numpy as np

# Ensure reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Constants
MODEL_NAME = "dbmdz/bert-base-multilingual-cased-finetuned-ner"
OUTPUT_FILE = "sustainability_insights.csv"

# Load Pre-trained Multilingual Model
def load_model():
    """Loads the tokenizer and model for NER."""
    print("Loading NER model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME)
    nlp_pipeline = pipeline("ner", model=model, tokenizer=tokenizer, grouped_entities=True)
    print("Model loaded successfully.")
    return nlp_pipeline

# Preprocess Reports
def preprocess_reports(texts):
    """Cleans and standardizes the text for consistent processing."""
    cleaned_texts = [text.replace("\n", " ").strip() for text in texts]
    print(f"Preprocessed {len(cleaned_texts)} reports.")
    return cleaned_texts

# Extract Entities
def extract_entities(nlp_pipeline, texts):
    """Extracts entities from the provided text using the NER pipeline."""
    all_entities = []
    for idx, text in enumerate(texts):
        print(f"Processing report {idx + 1}/{len(texts)}...")
        entities = nlp_pipeline(text)
        processed_entities = [
            {
                "entity": entity["entity_group"],
                "word": entity["word"],
                "score": round(entity["score"], 4),
                "start": entity["start"],
                "end": entity["end"]
            } for entity in entities
        ]
        all_entities.append(processed_entities)
    print("Entity extraction completed.")
    return all_entities

# Generate Insights
def generate_insights(entities_list):
    """Structures extracted entities into actionable insights."""
    insights = []
    for idx, entities in enumerate(entities_list):
        print(f"Generating insights for report {idx + 1}/{len(entities_list)}...")
        insight = {
            "Organizations": [e["word"] for e in entities if e["entity"] == "ORG"],
            "Locations": [e["word"] for e in entities if e["entity"] == "LOC"],
            "Dates": [e["word"] for e in entities if e["entity"] == "DATE"],
        }
        insights.append(insight)
    print("Insights generation completed.")
    return insights

# Save Insights to CSV
def save_to_csv(insights):
    """Saves the generated insights to a CSV file."""
    insights_df = pd.DataFrame(insights)
    insights_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Insights saved to {OUTPUT_FILE}.")

# Main Function
def main():
    # Example data: Replace with your sustainability reports
    sustainability_reports = [
        "Company ABC reduced emissions in 2023 by relocating operations to Berlin.",
        "Organization XYZ aims to be carbon neutral by 2030, with key operations in Paris."
    ]

    # Load model
    nlp_pipeline = load_model()

    # Preprocess reports
    preprocessed_reports = preprocess_reports(sustainability_reports)

    # Extract entities
    extracted_entities = extract_entities(nlp_pipeline, preprocessed_reports)

    # Generate insights
    insights = generate_insights(extracted_entities)

    # Save insights
    save_to_csv(insights)

if __name__ == "__main__":
    main()

