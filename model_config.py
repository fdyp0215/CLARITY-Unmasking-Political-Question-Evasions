from dataclasses import dataclass
from typing import Dict, List

@dataclass
class ModelConfig:
    # Model parameters
    model_name: str = "microsoft/deberta-v3-base"
    # model_name: str = "roberta-base"
    # model_name: str = "bert-base-uncased"
    max_length: int = 512
    dropout_rate: float = 0.2
    
    # Label mappings
    clarity_labels: List[str] = None
    evasion_labels: List[str] = None
    
    # Training parameters
    batch_size: int = 16
    learning_rate: float = 3e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1

    # 解冻的配置
    # unfreeze_last_n_layers: int = 8  # 解冻最后N层
    freeze_first_n_layers: int = 4  # 冻结前N层
    
    # Multi-task learning
    task_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.clarity_labels is None:
            self.clarity_labels = ['Clear Reply', 'Ambivalent', 'Clear Non-Reply']
        if self.evasion_labels is None:
            self.evasion_labels = ['Explicit', 'General', 'Partial/half-answer', 'Dodging', 
                                 'Implicit', 'Deflection', 'Declining to answer', 
                                 'Claims ignorance', 'Clarification']
        if self.task_weights is None:
            self.task_weights = {'clarity': 0.5, 'evasion': 0.5}
