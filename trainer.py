import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, f1_score, accuracy_score
import wandb

class MTLTrainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        clarity_preds = []
        clarity_labels = []
        evasion_preds = []
        evasion_labels = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            self.optimizer.zero_grad()
            
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            batch_clarity_labels = batch['clarity_labels'].to(self.device)
            batch_evasion_labels = batch['evasion_labels'].to(self.device)
            
            # Filter out invalid labels (-1)
            valid_indices = (batch_evasion_labels != -1) & (batch_clarity_labels != -1)
            
            if valid_indices.sum() == 0:
                continue
                
            # Only use valid samples
            input_ids = input_ids[valid_indices]
            attention_mask = attention_mask[valid_indices]
            batch_clarity_labels = batch_clarity_labels[valid_indices]
            batch_evasion_labels = batch_evasion_labels[valid_indices]
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                clarity_labels=batch_clarity_labels,
                evasion_labels=batch_evasion_labels
            )
            
            loss = outputs['loss']
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions
            clarity_pred = torch.argmax(outputs['clarity_logits'], dim=1)
            evasion_pred = torch.argmax(outputs['evasion_logits'], dim=1)
            
            clarity_preds.extend(clarity_pred.cpu().numpy())
            clarity_labels.extend(batch_clarity_labels.cpu().numpy())
            evasion_preds.extend(evasion_pred.cpu().numpy())
            evasion_labels.extend(batch_evasion_labels.cpu().numpy())
            
            pbar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        if clarity_labels and evasion_labels:
            clarity_f1 = f1_score(clarity_labels, clarity_preds, average='macro')
            clarity_acc = accuracy_score(clarity_labels, clarity_preds)
            evasion_f1 = f1_score(evasion_labels, evasion_preds, average='macro')
            evasion_acc = accuracy_score(evasion_labels, evasion_preds)
        else:
            clarity_f1 = clarity_acc = evasion_f1 = evasion_acc = 0.0
        
        return {
            'train_loss': total_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0,
            'clarity_f1': clarity_f1,
            'clarity_acc': clarity_acc,
            'evasion_f1': evasion_f1,
            'evasion_acc': evasion_acc
        }
    
    def _calculate_evasion_accuracy(self, preds, annotators):
        """计算evasion准确率：三个annotator有任意一个和预测的一样都算对"""
        correct = 0
        total = 0
        
        for pred, annotator_list in zip(preds, annotators):
            # 过滤掉-1（无效标签）
            valid_annotators = [ann for ann in annotator_list if ann != -1]
            if not valid_annotators:
                continue
                
            # 如果预测的标签在任意一个annotator的标签中，就算正确
            if pred in valid_annotators:
                correct += 1
            total += 1
        
        return correct / total if total > 0 else 0.0
    
    def _calculate_evasion_f1(self, preds, annotators, average='macro'):
        """计算evasion F1分数：使用多标签的方式"""
        from sklearn.preprocessing import MultiLabelBinarizer
        from sklearn.metrics import f1_score
        
        # 准备真实标签（多标签格式）
        y_true = []
        y_pred = []
        
        for pred, annotator_list in zip(preds, annotators):
            valid_annotators = [ann for ann in annotator_list if ann != -1]
            if not valid_annotators:
                continue
                
            y_true.append(valid_annotators)
            y_pred.append([pred])
        
        if not y_true:
            return 0.0
            
        # 使用多标签二值化
        mlb = MultiLabelBinarizer(classes=range(len(self.config.evasion_labels)))
        y_true_bin = mlb.fit_transform(y_true)
        y_pred_bin = mlb.transform(y_pred)
        
        return f1_score(y_true_bin, y_pred_bin, average=average)
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        clarity_preds = []
        clarity_labels = []
        evasion_preds = []
        evasion_annotators_list = []  # 存储所有annotator的标签
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                batch_clarity_labels = batch['clarity_labels'].to(self.device)
                batch_evasion_labels = batch['evasion_labels'].to(self.device)
                batch_evasion_annotators = batch['evasion_annotators'].to(self.device)
                
                # Filter out invalid labels (-1)
                valid_indices = (batch_evasion_labels != -1) & (batch_clarity_labels != -1)
                
                if valid_indices.sum() == 0:
                    continue
                    
                # Only use valid samples
                input_ids = input_ids[valid_indices]
                attention_mask = attention_mask[valid_indices]
                batch_clarity_labels = batch_clarity_labels[valid_indices]
                batch_evasion_labels = batch_evasion_labels[valid_indices]
                batch_evasion_annotators = batch_evasion_annotators[valid_indices]
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    clarity_labels=batch_clarity_labels,
                    evasion_labels=batch_evasion_labels
                )
                
                total_loss += outputs['loss'].item()
                
                clarity_pred = torch.argmax(outputs['clarity_logits'], dim=1)
                evasion_pred = torch.argmax(outputs['evasion_logits'], dim=1)
                
                clarity_preds.extend(clarity_pred.cpu().numpy())
                clarity_labels.extend(batch_clarity_labels.cpu().numpy())
                evasion_preds.extend(evasion_pred.cpu().numpy())
                evasion_annotators_list.extend(batch_evasion_annotators.cpu().numpy())
        
        # Calculate metrics
        if clarity_labels and evasion_preds:
            clarity_f1 = f1_score(clarity_labels, clarity_preds, average='macro')
            clarity_acc = accuracy_score(clarity_labels, clarity_preds)
            
            # 使用新的评估方式计算evasion指标
            evasion_acc = self._calculate_evasion_accuracy(evasion_preds, evasion_annotators_list)
            evasion_f1 = self._calculate_evasion_f1(evasion_preds, evasion_annotators_list, average='macro')
            
            # Print detailed classification reports
            print("\nClarity Classification Report:")
            print(classification_report(clarity_labels, clarity_preds, 
                                      target_names=self.config.clarity_labels))
            
            print("\nEvasion Classification Report:")
            print(f"acc: {evasion_acc:.4f}")
            print(f"F1分数 (macro f1): {evasion_f1:.4f}")
            
            # 显示详细的匹配情况
            self._print_evasion_details(evasion_preds, evasion_annotators_list)
            
        else:
            clarity_f1 = clarity_acc = evasion_f1 = evasion_acc = 0.0
            print("No valid samples for validation!")
        
        return {
            'val_loss': total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0,
            'clarity_f1': clarity_f1,
            'clarity_acc': clarity_acc,
            'evasion_f1': evasion_f1,
            'evasion_acc': evasion_acc
        }
    
    def _print_evasion_details(self, preds, annotators_list):
        """打印evasion任务的详细匹配情况"""
        match_counts = [0, 0, 0]  # 匹配1个、2个、3个annotator的数量
        total_valid = 0
        
        for pred, annotators in zip(preds, annotators_list):
            valid_annotators = [ann for ann in annotators if ann != -1]
            if not valid_annotators:
                continue
                
            total_valid += 1
            matches = sum(1 for ann in valid_annotators if ann == pred)
            if matches > 0:
                match_counts[min(matches-1, 2)] += 1
        
        print(f"\nEvasion 详细匹配情况 (总样本数: {total_valid}):")
        print(f"匹配1个annotator: {match_counts[0]} ({match_counts[0]/total_valid*100:.1f}%)")
        if len([x for x in match_counts if x > 1]) > 1:
            print(f"匹配2个annotator: {match_counts[1]} ({match_counts[1]/total_valid*100:.1f}%)")
        if len([x for x in match_counts if x > 2]) > 2:
            print(f"匹配3个annotator: {match_counts[2]} ({match_counts[2]/total_valid*100:.1f}%)")
    
    def train(self, num_epochs):
        best_val_f1 = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 50)
            
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                **{f'train_{k}': v for k, v in train_metrics.items()},
                **{f'val_{k}': v for k, v in val_metrics.items()}
            }
            
            print(f"Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"Val Clarity F1: {val_metrics['clarity_f1']:.4f}")
            print(f"Val Evasion F1: {val_metrics['evasion_f1']:.4f}")
            print(f"Val Evasion Acc : {val_metrics['evasion_acc']:.2f}")
            
            # Save best model
            if val_metrics['clarity_f1'] > 0 and val_metrics['evasion_f1'] > 0:
                avg_f1 = (val_metrics['clarity_f1'] + val_metrics['evasion_f1']) / 2
                if avg_f1 > best_val_f1:
                    best_val_f1 = avg_f1
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_val_f1': best_val_f1,
                        'config': self.config
                    }, 'best_model.pth')
                    print(f"Saved best model with avg F1: {best_val_f1:.4f}")
