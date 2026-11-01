"""
DIPCYT-X40s — Configuration
============================
Dataclass-backed configuration with YAML serialisation.
Every hyperparameter is exposed — baseline preprocessing, architecture,
loss weights, matching costs, and single-GPU hardware-safe runtime options.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional


@dataclass
class ModelConfig:
    img_size: int = 640
    num_queries: int = 32
    num_points: int = 6
    fpn_out_channels: int = 256
    pretrained_backbone: bool = True
    pretrained_cache_path: str = "yolov5s.pt"
    head_encoder_layers: List[int] = field(default_factory=lambda: [1, 2, 2, 2])
    head_decoder_layers: int = 6
    head_nhead: int = 8
    head_dim_feedforward: int = 1024
    head_dropout: float = 0.1
    conf_bias_init: float = -4.0
    fpcbam_reduction_ratio: int = 16
    swin_window_size: int = 5
    swin_nhead: int = 8
    enable_gradient_checkpointing: bool = False

@dataclass
class LossConfig:
    lambda_coord: float = 5.0
    lambda_giou: float = 2.0
    lambda_conf: float = 2.0
    lambda_shape: float = 1.0
    focal_alpha: float = 0.75
    focal_gamma: float = 1.0


@dataclass
class MatcherConfig:
    center_cost_weight: float = 5.0
    size_cost_weight: float = 1.0
    giou_cost_weight: float = 2.0
    bucket_thresholds: List[float] = field(
        default_factory=lambda: [0.08, 0.16, 0.30]
    )

@dataclass
class HardwareConfig:
    auto_probe_vram: bool = True
    vram_safety_margin: float = 0.20
    target_effective_batch_size: int = 8
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 2
    max_oom_retries: int = 3
    use_amp: bool = True
    amp_dtype: str = "float16"
    num_workers: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    log_file: str = "training.log"
    log_every_n_steps: int = 20
    keep_last_k_checkpoints: int = 3

@dataclass
class TrainConfig:
    seed: int = 42
    num_epochs: int = 30
    optimizer: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    sgd_momentum: float = 0.9
    scheduler: str = "cosine"
    warmup_steps: int = 0
    grad_clip: float = 1.0
    backbone_unfreeze_epoch: int = 0
    checkpoint_dir: str = "checkpoints"
    save_every_epoch: int = 1
    reset_optimizer: bool = False
    proxy_num_images: int = 10
    proxy_num_iters: int = 800
    proxy_log_every: int = 100


@dataclass
class DataConfig:
    img_dir: str = "train2014"
    ann_file: str = "cocotext.v2.json"
    max_targets: int = 128
    img_size: int = 640
    min_poly_area: float = 1e-6
    min_box_pixels: int = 26
    augment: bool = True
    scale_range_lo: float = 0.7
    scale_range_hi: float = 1.3
    rotate_max_angle: int = 10
    flip_prob: float = 0.5
    jitter_prob: float = 0.5
    blur_prob: float = 0.2
    rotate_prob: float = 0.4

@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    matcher: MatcherConfig = field(default_factory=MatcherConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_yaml(self, path: str | Path) -> None:
        """Serialise full config to YAML file."""
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Deserialise Config from YAML file."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f) or {}
        return cls(
            model=ModelConfig(**d.get("model", {})),
            loss=LossConfig(**d.get("loss", {})),
            matcher=MatcherConfig(**d.get("matcher", {})),
            hardware=HardwareConfig(**d.get("hardware", {})),
            train=TrainConfig(**d.get("train", {})),
            data=DataConfig(**d.get("data", {})),
        )

    def to_dict(self) -> dict:
        return asdict(self)
