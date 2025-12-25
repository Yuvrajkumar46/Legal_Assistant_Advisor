"""
Legal Assistant Model Training Script
This script trains a small transformer model for legal Q&A
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# Configuration
class Config:
    MODEL_NAME = "microsoft/DialoGPT-small"  # Small conversational model
    MAX_LENGTH = 512
    BATCH_SIZE = 4
    LEARNING_RATE = 5e-5
    EPOCHS = 3
    OUTPUT_DIR = "./legal_assistant_model"
    LOGGING_DIR = "./logs"

# Dataset class for legal Q&A
class LegalQADataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        question = item['question']
        answer = item['answer']
        
        # Format as conversation
        text = f"User: {question}\nAssistant: {answer}\n"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

def create_sample_dataset():
    with open('dataset.json', 'r') as f:
        data = json.load(f)

    expanded = []
    for item in data:
        q = item["question"]
        a = item["answer"]

        # Original
        expanded.append({"question": q, "answer": a})

        # Variation 1
        expanded.append({
            "question": "Can you explain " + q.lower(),
            "answer": a
        })

        # Variation 2
        expanded.append({
            "question": "Tell me about " + q.lower().replace("what", "").replace("how", "").replace("?", ""),
            "answer": a
        })

    return expanded

def train_model():
    """
    Main training function
    """
    print("Initializing model training...")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(Config.MODEL_NAME)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataset
    print("Creating dataset...")
    data = create_sample_dataset()
    
    # Split data
    train_data, val_data = train_test_split(data, test_size=0.1, random_state=42)
    
    # Create datasets
    train_dataset = LegalQADataset(train_data, tokenizer, Config.MAX_LENGTH)
    val_dataset = LegalQADataset(val_data, tokenizer, Config.MAX_LENGTH)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=Config.OUTPUT_DIR,
        num_train_epochs=Config.EPOCHS,
        per_device_train_batch_size=Config.BATCH_SIZE,
        per_device_eval_batch_size=Config.BATCH_SIZE,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir=Config.LOGGING_DIR,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        
        push_to_hub=False,
        report_to="none",  # Disable wandb/tensorboard for simplicity
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # We're doing causal LM, not masked LM
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {Config.OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(Config.OUTPUT_DIR)
    
    print("Training complete!")

def test_model(prompt):
    """
    Test the trained model with a prompt
    """
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.OUTPUT_DIR)
    model = AutoModelForCausalLM.from_pretrained(Config.OUTPUT_DIR)
    
    # Prepare input
    input_text = f"User: {prompt}\nAssistant:"
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            inputs, 
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract assistant's response
    if "Assistant:" in response:
        response = response.split("Assistant:")[-1].strip()
    
    return response

if __name__ == "__main__":
    # Train the model
    train_model()
    
    # Test with a sample question
    print("\n" + "="*50)
    print("Testing the model...")
    print("="*50)
    
    test_prompts = [
        "What is a contract?",
        "How do I protect my intellectual property?",
        "What are my rights if I get arrested?"
    ]
    
    for prompt in test_prompts:
        print(f"\nUser: {prompt}")
        response = test_model(prompt)
        print(f"Assistant: {response}")