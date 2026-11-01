"""
DIPCYT-X40s Pipeline Package
"""

from .config import Config, ModelConfig, LossConfig, MatcherConfig, TrainConfig, DataConfig
from .model import FullModel, TPHYOLOv5PlusPlus, FPCBAM, PANet, TransformerPolygonHead
from .dataset import COCOTextDataset, COCOTextAnnotationStore, build_loaders, polygon_collate_fn
from .losses import PolygonHungarianLoss
from .eval import evaluate_model, evaluate_instance_metrics, evaluate_area_metric, sweep_confidence_thresholds, polygon_iou
from .train import train_pipeline, run_proxy_trial, save_checkpoint, load_checkpoint

try:
    from .optuna_search import run_optuna_search
except ImportError:
    run_optuna_search = None

__all__ = [
    "Config",
    "ModelConfig",
    "LossConfig",
    "MatcherConfig",
    "TrainConfig",
    "DataConfig",
    "FullModel",
    "TPHYOLOv5PlusPlus",
    "FPCBAM",
    "PANet",
    "TransformerPolygonHead",
    "COCOTextDataset",
    "COCOTextAnnotationStore",
    "build_loaders",
    "polygon_collate_fn",
    "PolygonHungarianLoss",
    "evaluate_model",
    "evaluate_instance_metrics",
    "evaluate_area_metric",
    "sweep_confidence_thresholds",
    "polygon_iou",
    "train_pipeline",
    "run_proxy_trial",
    "save_checkpoint",
    "load_checkpoint",
    "run_optuna_search",
]
