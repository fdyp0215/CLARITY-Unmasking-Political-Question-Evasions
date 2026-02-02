import torch
import pandas as pd
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_config import ModelConfig
from data_loader import create_data_loaders
from deberta_mtl import DebertaMTL
from roberta_mtl import RobertaMTL
from bert_mtl import BertMTL
from trainer import MTLTrainer

def main():
    # Configuration
    config = ModelConfig()
    
    # Load data
    print("Loading data...")
    splits = {'train': 'data/train-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
    df_train = pd.read_parquet("hf://datasets/ailsntua/QEvasion/" + splits["train"])
    df_test = pd.read_parquet("hf://datasets/ailsntua/QEvasion/" + splits["test"])
    
    # Create data loaders
    train_loader, test_loader, tokenizer = create_data_loaders(df_train, df_test, config, use_augmentation=True)
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Model setup 
    print("Initializing model...")
    model = DebertaMTL(config.model_name, config)  # 直接初始化，不使用from_pretrained
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    # Trainer
    trainer = MTLTrainer(model, train_loader, test_loader, optimizer, device, config)
    
    # Training
    print("Starting training...")
    trainer.train(config.num_epochs)
    
    print("Training completed!")

if __name__ == "__main__":
    main()