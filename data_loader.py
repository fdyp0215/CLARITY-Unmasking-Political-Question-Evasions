import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch
from model_config import ModelConfig
import numpy as np

class PoliticalDiscourseDataset(Dataset):
    def __init__(self, df, tokenizer, config, is_train=True, use_augmentation=True):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.max_length
        self.is_train = is_train
        self.use_augmentation = use_augmentation and is_train  # 只在训练时使用增强
        
        # Create label mappings
        self.clarity_label2id = {label: idx for idx, label in enumerate(config.clarity_labels)}
        self.evasion_label2id = {label: idx for idx, label in enumerate(config.evasion_labels)}
        
    def __len__(self):
        if self.use_augmentation and self.is_train:
            # 如果使用增强，数据量翻倍（原始数据 + 增强数据）
            return len(self.df) * 2
        return len(self.df)
    
    def __getitem__(self, idx):
        # 判断是否使用增强数据
        use_augmented = False
        if self.use_augmentation and self.is_train:
            # 前半部分用原始数据，后半部分用增强数据
            if idx >= len(self.df):
                use_augmented = True
                original_idx = idx - len(self.df)
            else:
                original_idx = idx
        else:
            original_idx = idx
        
        row = self.df.iloc[original_idx]
        
        # 选择文本来源
        if use_augmented:
            # 使用GPT摘要作为增强数据
            if 'gpt3.5_summary' in row and pd.notna(row['gpt3.5_summary']) and row['gpt3.5_summary'].strip():
                text = f"Summary: {row['gpt3.5_summary']}"
            else:
                # 如果没有摘要，回退到原始文本
                text = f"Question: {row['question']} Answer: {row['interview_answer']}"
        else:
            # 使用原始文本
            text = f"Question: {row['question']} Answer: {row['interview_answer']}"
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Get clarity label (same for both train and test)
        clarity_label = self.clarity_label2id.get(row['clarity_label'], -1)
        
        # Handle evasion label differently for train vs test
        if self.is_train:
            # For training data, use evasion_label column
            evasion_label = self.evasion_label2id.get(row['evasion_label'], -1)
            # For training, we still use single label, but store all annotators for validation
            evasion_annotators = torch.tensor([evasion_label], dtype=torch.long)
        else:
            # For test data, collect all three annotators' labels
            annotator_labels = []
            for i in range(1, 4):  # annotator1, annotator2, annotator3
                annotator_col = f'annotator{i}'
                if annotator_col in row and not pd.isna(row[annotator_col]) and row[annotator_col] != '':
                    label_id = self.evasion_label2id.get(row[annotator_col], -1)
                    if label_id != -1:
                        annotator_labels.append(label_id)
            
            # Use the first annotator as the main label for compatibility
            evasion_label = annotator_labels[0] if annotator_labels else -1
            evasion_annotators = torch.tensor(annotator_labels, dtype=torch.long) if annotator_labels else torch.tensor([-1], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'clarity_labels': torch.tensor(clarity_label, dtype=torch.long),
            'evasion_labels': torch.tensor(evasion_label, dtype=torch.long),
            'evasion_annotators': evasion_annotators  # 存储所有annotator的标签
        }

def create_data_loaders(train_df, test_df, config, use_augmentation=True):
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    
    # 检查训练数据中是否有GPT摘要列
    if use_augmentation and 'gpt3.5_summary' not in train_df.columns:
        print("Warning: 'gpt3.5_summary' column not found in training data. Disabling augmentation.")
        use_augmentation = False
    
    # 检查GPT摘要列的质量
    if use_augmentation:
        valid_summaries = train_df['gpt3.5_summary'].notna() & (train_df['gpt3.5_summary'].str.strip() != '')
        valid_count = valid_summaries.sum()
        print(f"GPT摘要数据质量: {valid_count}/{len(train_df)} 行有有效摘要")
    
    # Create datasets
    train_dataset = PoliticalDiscourseDataset(
        train_df, tokenizer, config, is_train=True, use_augmentation=use_augmentation
    )
    test_dataset = PoliticalDiscourseDataset(
        test_df, tokenizer, config, is_train=False, use_augmentation=False
    )
    
    print(f"训练数据大小: {len(train_dataset)} (原始: {len(train_df)}, 增强: {len(train_dataset) - len(train_df)})")
    print(f"测试数据大小: {len(test_dataset)}")
    
    # 自定义collate函数来处理可变长度的evasion_annotators
    def collate_fn(batch):
        batch_size = len(batch)
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        clarity_labels = torch.stack([item['clarity_labels'] for item in batch])
        evasion_labels = torch.stack([item['evasion_labels'] for item in batch])
        
        # 处理可变长度的evasion_annotators
        max_annotators = max(item['evasion_annotators'].size(0) for item in batch)
        evasion_annotators_padded = torch.full((batch_size, max_annotators), -1, dtype=torch.long)
        
        for i, item in enumerate(batch):
            annotators = item['evasion_annotators']
            evasion_annotators_padded[i, :annotators.size(0)] = annotators
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'clarity_labels': clarity_labels,
            'evasion_labels': evasion_labels,
            'evasion_annotators': evasion_annotators_padded
        }
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    return train_loader, test_loader, tokenizer