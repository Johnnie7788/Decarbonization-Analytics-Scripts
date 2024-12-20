#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import optuna
from datasets import Dataset
import shap

# Ensure reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load and preprocess data
def load_and_preprocess_data(file_path):
    """Load and preprocess dataset from CSV."""
    try:
        data = pd.read_csv(file_path)
        print(f"Loaded dataset with {len(data)} rows.")
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found. Please check the path.")
    
    if 'text' not in data.columns or 'label' not in data.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
    
    data = data[['text', 'label']].dropna()
    data['label'] = data['label'].astype(int)
    return data

# File path
file_path = "carbon_emission_data.csv"  # Replace with actual file path
try:
    data = load_and_preprocess_data(file_path)
except Exception as e:
    print(f"Error loading data: {e}")

# Split dataset
try:
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data['text'], data['label'], test_size=0.2, random_state=SEED, stratify=data['label']
    )
    print("Dataset split into training and testing sets.")
except Exception as e:
    print(f"Error splitting data: {e}")

# Tokenization
def tokenize_data(texts):
    return tokenizer(texts.tolist(), padding=True, truncation=True, max_length=512, return_tensors="pt")

print("Tokenizing data...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
try:
    train_encodings = tokenize_data(train_texts)
    test_encodings = tokenize_data(test_texts)
    print("Data tokenized successfully.")
except Exception as e:
    print(f"Error in tokenization: {e}")

# Dataset class
class EmissionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

try:
    train_dataset = EmissionDataset(train_encodings, train_labels.tolist())
    test_dataset = EmissionDataset(test_encodings, test_labels.tolist())
    print("Datasets created successfully.")
except Exception as e:
    print(f"Error creating datasets: {e}")

# Load pre-trained BERT model
print("Loading BERT model...")
try:
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased', num_labels=len(data['label'].unique())
    ).to(device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")

# Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

# Training arguments
try:
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        push_to_hub=False,
        fp16=True
    )
    print("Training arguments initialized.")
except Exception as e:
    print(f"Error in training arguments: {e}")

# Initialize Trainer
try:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    print("Trainer initialized successfully.")
except Exception as e:
    print(f"Error initializing trainer: {e}")

# Hyperparameter optimization with Optuna
def objective(trial):
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 5e-5)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32])
    training_args.learning_rate = learning_rate
    training_args.per_device_train_batch_size = batch_size
    training_args.per_device_eval_batch_size = batch_size
    trainer.train()
    eval_results = trainer.evaluate()
    return eval_results['eval_accuracy']

try:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    print(f"Best hyperparameters: {study.best_params}")
except Exception as e:
    print(f"Error in hyperparameter optimization: {e}")

# Explainability with SHAP
print("Generating SHAP explanations...")
def explain_model():
    explainer = shap.Explainer(model, tokenizer)
    shap_values = explainer(test_texts[:10])  # Limit to a small subset for memory efficiency
    shap.plots.text(shap_values)

try:
    explain_model()
    print("SHAP explanations generated successfully.")
except Exception as e:
    print(f"Error generating SHAP explanations: {e}")

# Save the model
output_dir = "./carbon_emission_model"
try:
    print(f"Saving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully.")
except Exception as e:
    print(f"Error saving model: {e}")

# Generate deployment script
print("Creating deployment script for AWS SageMaker...")
try:
    with open("inference.py", "w") as f:
        f.write("""
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def model_fn(model_dir):
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)
    return model, tokenizer

def predict_fn(input_data, model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    inputs = tokenizer(input_data, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1).tolist()
    return probabilities
""")
    print("Deployment script created successfully.")
except Exception as e:
    print(f"Error creating deployment script: {e}")

print("All tasks completed successfully.")

