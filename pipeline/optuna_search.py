"""
DIPCYT-X40s — Automated Hyperparameter Optimization via Optuna
===============================================================
Sampler: Tree-structured Parzen Estimator (TPE).
  Citation: Bergstra et al., "Algorithms for Hyper-Parameter Optimization", NeurIPS 2011.

Pruner: Hyperband (with fallback to MedianPruner if early loss curves are too volatile).
  Citation: Li et al., "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization", JMLR 2018.

Objective: Validation F1 / H-mean on proxy progressive-overfit dataset (N=10 images, 800 iters).
Reproducibility: Fixed seed per trial ensures strict deterministic search without seed confounding.
Single-GPU Safety: Integrated VRAM probe filters out unfeasible micro-batch dimensions.
Exports: best_params.yaml + study_trials.csv.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Optional, Dict, Any, List

import optuna
from optuna.pruners import HyperbandPruner, MedianPruner
from optuna.samplers import TPESampler
import torch

from .config import Config, ModelConfig, LossConfig, MatcherConfig, TrainConfig, HardwareConfig
from .train import run_proxy_trial, probe_vram_and_batch_size, PipelineLogger
from .model import FullModel


# ── Objective Function Builder ───────────────────────────────────────────────

def make_objective(base_cfg: Config, feasible_batches: List[int]):
    def objective(trial: optuna.Trial) -> float:
        cfg = copy.deepcopy(base_cfg)

        # ── 1. Optimizer & Learning Rate ──
        cfg.train.optimizer = trial.suggest_categorical("optimizer", ["AdamW", "SGD"])
        cfg.train.lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        cfg.train.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        if cfg.train.optimizer == "SGD":
            cfg.train.sgd_momentum = trial.suggest_float("sgd_momentum", 0.85, 0.99)

        # ── 2. Learning Rate Scheduler ──
        cfg.train.scheduler = trial.suggest_categorical("scheduler", ["cosine", "linear_warmup_decay"])
        cfg.train.warmup_steps = trial.suggest_int("warmup_steps", 0, 200, step=20)

        # ── 3. Batch Size (GPU VRAM Feasible Dimensions Only) ──
        cfg.hardware.micro_batch_size = trial.suggest_categorical("batch_size", feasible_batches)

        # ── 4. Loss Component Weights (Independent Dimensions) ──
        cfg.loss.focal_alpha = trial.suggest_float("focal_alpha", 0.25, 0.90)
        cfg.loss.focal_gamma = trial.suggest_float("focal_gamma", 0.5, 3.0)
        cfg.loss.lambda_giou = trial.suggest_float("lambda_giou", 0.5, 5.0)
        cfg.loss.lambda_shape = trial.suggest_float("lambda_shape", 0.1, 3.0)

        # ── 5. Hungarian Cost Coefficients ──
        cfg.matcher.center_cost_weight = trial.suggest_float("center_cost_weight", 1.0, 10.0)
        cfg.matcher.size_cost_weight = trial.suggest_float("size_cost_weight", 0.2, 5.0)
        cfg.matcher.giou_cost_weight = trial.suggest_float("giou_cost_weight", 0.5, 5.0)

        # ── 6. FPCBAM Reduction Ratio ──
        cfg.model.fpcbam_reduction_ratio = trial.suggest_categorical("fpcbam_reduction_ratio", [8, 16, 32])

        # ── 7. Backbone Unfreeze Epoch ──
        cfg.train.backbone_unfreeze_epoch = trial.suggest_int("backbone_unfreeze_epoch", 0, 10)

        # ── Execute Proxy Trial ──
        score = run_proxy_trial(cfg=cfg, trial=trial)
        return score

    return objective


# ── Study Runner ─────────────────────────────────────────────────────────────

def run_optuna_search(
    base_cfg: Config,
    n_trials: int = 50,
    timeout: Optional[int] = None,
    study_name: str = "dipcyt_x40s_hpo",
    storage: Optional[str] = None,
    pruner_type: str = "hyperband",
    output_dir: str | Path = "optuna_results",
) -> optuna.Study:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    logger = PipelineLogger(out_path / "optuna.log")

    # 1. Probe feasible batch sizes on single GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feasible_batches = [2, 4, 8]
    if device.type == "cuda" and base_cfg.hardware.auto_probe_vram:
        temp_model = FullModel(base_cfg.model).to(device)
        max_safe_bs, _ = probe_vram_and_batch_size(temp_model, base_cfg, device, logger)
        feasible_batches = [b for b in [1, 2, 4, 8, 16] if b <= max_safe_bs]
        if not feasible_batches:
            feasible_batches = [1]
        del temp_model
        torch.cuda.empty_cache()

    # 2. Sampler: Tree-structured Parzen Estimator (TPE, Bergstra et al., NeurIPS 2011)
    sampler = TPESampler(seed=base_cfg.train.seed, multivariate=True)

    # 3. Pruner: Hyperband (Li et al., JMLR 2018) / MedianPruner fallback
    if pruner_type.lower() == "median":
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=100)
    else:
        pruner = HyperbandPruner(
            min_resource=100,
            max_resource=base_cfg.train.proxy_num_iters,
            reduction_factor=3,
        )

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )

    objective = make_objective(base_cfg=base_cfg, feasible_batches=feasible_batches)

    logger.log("=" * 65)
    logger.log(f"Starting Optuna HPO Study '{study_name}' ({n_trials} trials)")
    logger.log(f"Sampler: Tree-structured Parzen Estimator (TPE, Bergstra et al., 2011)")
    logger.log(f"Pruner : {pruner.__class__.__name__} (Li et al., 2018)")
    logger.log(f"Feasible Batch Sizes: {feasible_batches}")
    logger.log("=" * 65)

    try:
        study.optimize(objective, n_trials=n_trials, timeout=timeout)
    except KeyboardInterrupt:
        logger.log("\nOptimization manually interrupted by user.")

    logger.log("=" * 65)
    logger.log(f"Optuna HPO Complete. Best Trial: #{study.best_trial.number}")
    logger.log(f"Best Validation F1/H-mean: {study.best_value:.4f}")
    logger.log("Best Parameters:")
    for k, v in study.best_params.items():
        logger.log(f"  {k:<25}: {v}")
    logger.log("=" * 65)

    # ── Export Best Params as YAML ──
    best_cfg = copy.deepcopy(base_cfg)
    bp = study.best_params

    if "optimizer" in bp:
        best_cfg.train.optimizer = bp["optimizer"]
    if "lr" in bp:
        best_cfg.train.lr = bp["lr"]
    if "weight_decay" in bp:
        best_cfg.train.weight_decay = bp["weight_decay"]
    if "sgd_momentum" in bp:
        best_cfg.train.sgd_momentum = bp["sgd_momentum"]
    if "scheduler" in bp:
        best_cfg.train.scheduler = bp["scheduler"]
    if "warmup_steps" in bp:
        best_cfg.train.warmup_steps = bp["warmup_steps"]
    if "batch_size" in bp:
        best_cfg.hardware.micro_batch_size = bp["batch_size"]
    if "focal_alpha" in bp:
        best_cfg.loss.focal_alpha = bp["focal_alpha"]
    if "focal_gamma" in bp:
        best_cfg.loss.focal_gamma = bp["focal_gamma"]
    if "lambda_giou" in bp:
        best_cfg.loss.lambda_giou = bp["lambda_giou"]
    if "lambda_shape" in bp:
        best_cfg.loss.lambda_shape = bp["lambda_shape"]
    if "center_cost_weight" in bp:
        best_cfg.matcher.center_cost_weight = bp["center_cost_weight"]
    if "size_cost_weight" in bp:
        best_cfg.matcher.size_cost_weight = bp["size_cost_weight"]
    if "giou_cost_weight" in bp:
        best_cfg.matcher.giou_cost_weight = bp["giou_cost_weight"]
    if "fpcbam_reduction_ratio" in bp:
        best_cfg.model.fpcbam_reduction_ratio = bp["fpcbam_reduction_ratio"]
    if "backbone_unfreeze_epoch" in bp:
        best_cfg.train.backbone_unfreeze_epoch = bp["backbone_unfreeze_epoch"]

    best_yaml_path = out_path / "best_params.yaml"
    best_cfg.to_yaml(best_yaml_path)
    logger.log(f"✓ Saved winning configuration → {best_yaml_path}")

    # ── Export Full Study DataFrame as CSV ──
    df = study.trials_dataframe()
    csv_path = out_path / "study_trials.csv"
    df.to_csv(csv_path, index=False)
    logger.log(f"✓ Exported {len(df)} trials to CSV → {csv_path}")

    return study


# ── CLI Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIPCYT-X40s Optuna HPO Search")
    parser.add_argument("--config", type=str, default=None, help="Base config YAML")
    parser.add_argument("--n-trials", type=int, default=30, help="Number of trials")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    parser.add_argument("--pruner", type=str, default="hyperband", choices=["hyperband", "median"])
    parser.add_argument("--out-dir", type=str, default="optuna_results")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    run_optuna_search(
        base_cfg=cfg,
        n_trials=args.n_trials,
        timeout=args.timeout,
        pruner_type=args.pruner,
        output_dir=args.out_dir,
    )
