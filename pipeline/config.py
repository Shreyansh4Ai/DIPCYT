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


# ── Model Architecture ───────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    img_size: int = 640
    num_queries: int = 32          # queries per head (4 heads = 128 total)
    num_points: int = 6            # 6 polygon nodes for scene text (Roy et al., Sec 3.3)
    fpn_out_channels: int = 256
    pretrained_backbone: bool = True
    pretrained_cache_path: str = "yolov5s.pt"
    # Head 0 (40×40) gets 1 encoder layer for VRAM memory safety at highest resolution;
    # heads 1-3 (20×20, 10×10, 5×5) get 2 encoder layers.
    head_encoder_layers: List[int] = field(default_factory=lambda: [1, 2, 2, 2])
    head_decoder_layers: int = 6
    head_nhead: int = 8
    head_dim_feedforward: int = 1024
    head_dropout: float = 0.1
    conf_bias_init: float = -4.0
    # FPCBAM
    fpcbam_reduction_ratio: int = 16
    # Swin Transformer blocks inside PANet
    swin_window_size: int = 5
    swin_nhead: int = 8
    # Gradient Checkpointing (VRAM fallback for laptop GPUs)
    enable_gradient_checkpointing: bool = False


# ── Loss Configuration ───────────────────────────────────────────────────────

@dataclass
class LossConfig:
    lambda_coord: float = 5.0
    lambda_giou: float = 2.0
    lambda_conf: float = 2.0
    lambda_shape: float = 1.0
    focal_alpha: float = 0.75
    focal_gamma: float = 1.0


# ── Hungarian Matcher Configuration ──────────────────────────────────────────

@dataclass
class MatcherConfig:
    center_cost_weight: float = 5.0
    size_cost_weight: float = 1.0
    giou_cost_weight: float = 2.0
    # 4-head size bucket thresholds (normalised max(w,h) of GT box):
    # head 0 (D1, 40×40): max_size < 0.08 (smallest text)
    # head 1 (D2, 20×20): 0.08 <= max_size < 0.16
    # head 2 (D3, 10×10): 0.16 <= max_size < 0.30
    # head 3 (D4, 5×5):   max_size >= 0.30 (largest text)
    bucket_thresholds: List[float] = field(
        default_factory=lambda: [0.08, 0.16, 0.30]
    )


# ── Hardware & Long-Run Training Safety ───────────────────────────────────────

@dataclass
class HardwareConfig:
    # Single-GPU & VRAM Safety
    auto_probe_vram: bool = True               # Probe maximum micro-batch before training
    vram_safety_margin: float = 0.20           # 20% VRAM headroom margin
    target_effective_batch_size: int = 8       # Target batch size via grad accumulation
    micro_batch_size: int = 4                  # Starting micro-batch size (auto-adjusted by probe)
    gradient_accumulation_steps: int = 2       # Auto-computed: target // micro_batch_size
    max_oom_retries: int = 3                   # Retries per step upon CUDA OOM before halving bs
    # Mixed Precision (AMP)
    use_amp: bool = True
    amp_dtype: str = "float16"                 # "float16" or "bfloat16" (if GPU supports bf16)
    # CPU & Workers
    num_workers: int = 2                       # Tuned to laptop CPU cores (auto-capped)
    persistent_workers: bool = True            # Prevent worker respawn overhead
    pin_memory: bool = True                    # Fast host-to-device memory copy
    # Logging & Disk Usage Cap
    log_file: str = "training.log"             # Persistent disk log for unattended overnight runs
    log_every_n_steps: int = 20
    keep_last_k_checkpoints: int = 3           # Disk usage cap: retain only last k + best_model


# ── Training Loop Configuration ──────────────────────────────────────────────

@dataclass
class TrainConfig:
    seed: int = 42
    num_epochs: int = 30
    # Optimiser
    optimizer: str = "AdamW"                   # "AdamW" or "SGD"
    lr: float = 1e-4
    weight_decay: float = 1e-4
    sgd_momentum: float = 0.9
    # Scheduler
    scheduler: str = "cosine"                  # "cosine" or "linear_warmup_decay"
    warmup_steps: int = 0
    # Gradient clipping & Backbone Unfreezing
    grad_clip: float = 1.0
    backbone_unfreeze_epoch: int = 0           # 0 = always unfrozen
    # Checkpoint paths
    checkpoint_dir: str = "checkpoints"
    save_every_epoch: int = 1
    reset_optimizer: bool = False
    # Proxy training budget (Progressive Overfit for Optuna HPO)
    proxy_num_images: int = 10
    proxy_num_iters: int = 800
    proxy_log_every: int = 100


# ── Data & Baseline-Matched Preprocessing ────────────────────────────────────

@dataclass
class DataConfig:
    img_dir: str = "train2014"
    ann_file: str = "cocotext.v2.json"
    max_targets: int = 128
    
    # ── BASELINE-MATCHED PREPROCESSING (Roy et al., Pattern Recognition 2024) ──
    # [PDF Sec 4.2]: "All images are reshaped to 640 × 640 for training and testing."
    img_size: int = 640
    # [PDF Sec 4.2]: Image normalization with ImageNet standard mean and std.
    # [PDF Sec 3.3]: Resample polygon contours uniformly to 6 boundary points.
    min_poly_area: float = 1e-6
    # [PDF Sec 4.2 / TPH-YOLOv5++ Baseline]: Text instances with bounding box < 26px
    # cannot be resolved by 40×40 stride pyramid levels and are filtered.
    min_box_pixels: int = 26
    
    # ── AUGMENTATION POLICY ──
    # NOTE: The PDF states model configurations follow TPH-YOLOv5++ [11] (pre-train on COCO,
    # fine-tune on target datasets), but does not explicitly formulate custom crop/jitter ranges.
    # [ASSUMPTION]: Defaulting to COCO-Text / TPH-YOLOv5 standard geometric & color transforms.
    augment: bool = True
    scale_range_lo: float = 0.7
    scale_range_hi: float = 1.3
    rotate_max_angle: int = 10
    flip_prob: float = 0.5
    jitter_prob: float = 0.5
    blur_prob: float = 0.2
    rotate_prob: float = 0.4


# ── Top-Level Combined Configuration ─────────────────────────────────────────

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
