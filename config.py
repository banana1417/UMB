from dataclasses import dataclass


@dataclass
class UMBConfig:
    text_model_name: str = "roberta-base"
    vision_model_name: str = "google/vit-base-patch16-224"
    hidden_size: int = 768
    cross_modal_heads: int = 8
    umb_heads: int = 8
    dropout: float = 0.1
    eps: float = 1e-6
    pred_variance_floor: float = 0.1
    contrastive_temperature: float = 0.3
    contrastive_weight: float = 0.10
    reparameterization_samples: int = 7
    max_text_length: int = 128
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 1e-2
    max_epochs: int = 20
    patience: int = 5
    num_workers: int = 4
    seed: int = 42
    freeze_backbones: bool = False
