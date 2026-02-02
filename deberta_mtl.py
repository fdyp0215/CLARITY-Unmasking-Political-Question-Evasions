import torch
import torch.nn as nn
from transformers import DebertaV2Model, DebertaV2Config

class DebertaMTL(nn.Module):
    def __init__(self, model_name, model_config):
        super().__init__()
        # 加载预训练模型
        self.deberta = DebertaV2Model.from_pretrained(model_name)
        self.config = model_config
        
        # 冻结部分层，解冻部分层
        self._freeze_layers()
        
        self.dropout = nn.Dropout(model_config.dropout_rate)
        
        # Classification heads
        self.clarity_classifier = nn.Linear(self.deberta.config.hidden_size, len(model_config.clarity_labels))
        self.evasion_classifier = nn.Linear(self.deberta.config.hidden_size, len(model_config.evasion_labels))
        
        # Loss functions
        self.clarity_loss_fn = nn.CrossEntropyLoss()
        self.evasion_loss_fn = nn.CrossEntropyLoss()
        
        self.task_weights = model_config.task_weights
    
    def _freeze_layers(self):
        """冻结前N层，解冻后面的层"""
        total_layers = len(self.deberta.encoder.layer)
        
        # 配置解冻策略
        if hasattr(self.config, 'unfreeze_last_n_layers'):
            # 方案1: 解冻最后N层
            freeze_until_layer = total_layers - self.config.unfreeze_last_n_layers
        elif hasattr(self.config, 'freeze_first_n_layers'):
            # 方案2: 冻结前N层
            freeze_until_layer = self.config.freeze_first_n_layers
        else:
            # 默认: 冻结前8层，解冻后4层
            freeze_until_layer = 8
        
        print(f"Total layers: {total_layers}")
        print(f"Freezing first {freeze_until_layer} layers, unfreezing last {total_layers - freeze_until_layer} layers")
        
        # 冻结embeddings
        for param in self.deberta.embeddings.parameters():
            param.requires_grad = False
        
        # 冻结编码器层
        for i, layer in enumerate(self.deberta.encoder.layer):
            if i < freeze_until_layer:
                # 冻结前N层
                for param in layer.parameters():
                    param.requires_grad = False
            else:
                # 解冻后N层
                for param in layer.parameters():
                    param.requires_grad = True
                
        # 打印可训练参数统计
        self._print_trainable_parameters()
    
    def _print_trainable_parameters(self):
        """打印可训练参数统计"""
        total_params = 0
        trainable_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
                print(f"Trainable: {name}")
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Trainable ratio: {100 * trainable_params / total_params:.2f}%")
    
    def forward(self, input_ids, attention_mask=None, clarity_labels=None, evasion_labels=None):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True  # 确保输出hidden states
        )
        
        # 使用平均池化
        last_hidden_state = outputs.last_hidden_state
        pooled_output = torch.mean(last_hidden_state, dim=1)
        pooled_output = self.dropout(pooled_output)
        
        # Task-specific outputs
        clarity_logits = self.clarity_classifier(pooled_output)
        evasion_logits = self.evasion_classifier(pooled_output)
        
        loss = 0
        if clarity_labels is not None and evasion_labels is not None:
            clarity_loss = self.clarity_loss_fn(clarity_logits, clarity_labels)
            evasion_loss = self.evasion_loss_fn(evasion_logits, evasion_labels)
            
            loss = (self.task_weights['clarity'] * clarity_loss + 
                   self.task_weights['evasion'] * evasion_loss)
        
        return {
            'loss': loss,
            'clarity_logits': clarity_logits,
            'evasion_logits': evasion_logits,
            'hidden_states': outputs.hidden_states
        }