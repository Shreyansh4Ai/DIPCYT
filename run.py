"""
DIPCYT-X40s — Unified CLI Entry Point
======================================
Usage:
  # 1. Full Production Training (Single-GPU Laptop Safe):
  python run.py train --config default_config.yaml

  # 2. Automated Hyperparameter Optimization via Optuna (TPE + Hyperband):
  python run.py optuna --config default_config.yaml --n-trials 50 --out-dir optuna_results

  # 3. Model Evaluation on Validation / Test Set:
  python run.py eval --checkpoint checkpoints/best_model.pth --config default_config.yaml

  # 4. Confidence Threshold Sweep for Peak F1:
  python run.py sweep --checkpoint checkpoints/best_model.pth --config default_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch

from pipeline.config import Config
from pipeline.dataset import build_loaders
from pipeline.eval import evaluate_model, sweep_confidence_thresholds
from pipeline.model import FullModel
from pipeline.optuna_search import run_optuna_search
from pipeline.train import train_pipeline, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="DIPCYT-X40s Pipeline Runner")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Command to execute")

    # ── Train Subcommand ──
    train_parser = subparsers.add_parser("train", help="Run full model training with VRAM & OOM safety")
    train_parser.add_argument("--config", type=str, default="default_config.yaml", help="Path to config YAML")
    train_parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")

    # ── Optuna Subcommand ──
    optuna_parser = subparsers.add_parser("optuna", help="Run Optuna automated hyperparameter search")
    optuna_parser.add_argument("--config", type=str, default="default_config.yaml", help="Base config YAML")
    optuna_parser.add_argument("--n-trials", type=int, default=30, help="Number of HPO trials")
    optuna_parser.add_argument("--timeout", type=int, default=None, help="Timeout in seconds")
    optuna_parser.add_argument("--pruner", type=str, default="hyperband", choices=["hyperband", "median"])
    optuna_parser.add_argument("--out-dir", type=str, default="optuna_results", help="Directory for YAML/CSV exports")

    # ── Eval Subcommand ──
    eval_parser = subparsers.add_parser("eval", help="Evaluate checkpoint on validation/test set")
    eval_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    eval_parser.add_argument("--config", type=str, default="default_config.yaml", help="Path to config YAML")
    eval_parser.add_argument("--conf-thresh", type=float, default=0.3, help="Confidence threshold")
    eval_parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")

    # ── Sweep Subcommand ──
    sweep_parser = subparsers.add_parser("sweep", help="Sweep confidence thresholds for optimal F1")
    sweep_parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    sweep_parser.add_argument("--config", type=str, default="default_config.yaml", help="Path to config YAML")
    sweep_parser.add_argument("--split", type=str, default="val", choices=["val", "test"], help="Dataset split")

    args = parser.parse_args()

    # Load configuration
    cfg_path = Path(args.config)
    cfg = Config.from_yaml(cfg_path) if cfg_path.exists() else Config()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.command == "train":
        print(f"Starting training with config: {cfg_path}")
        train_pipeline(cfg, resume_path=args.resume)

    elif args.command == "optuna":
        print(f"Starting Optuna search with {args.n_trials} trials...")
        run_optuna_search(
            base_cfg=cfg,
            n_trials=args.n_trials,
            timeout=args.timeout,
            pruner_type=args.pruner,
            output_dir=args.out_dir,
        )

    elif args.command == "eval":
        print(f"Evaluating checkpoint: {args.checkpoint} on {args.split} split...")
        model = FullModel(cfg.model).to(device)
        load_checkpoint(model, path=args.checkpoint, device=device)
        _, val_loader, test_loader, _ = build_loaders(cfg)
        loader = val_loader if args.split == "val" else test_loader

        metrics = evaluate_model(model, loader, device, conf_thresh=args.conf_thresh)
        print("=" * 50)
        print(f"Results (Confidence > {args.conf_thresh}):")
        print(f"  Instance Precision : {metrics['precision']:.4f}")
        print(f"  Instance Recall    : {metrics['recall']:.4f}")
        print(f"  Instance F1        : {metrics['f1']:.4f}")
        print(f"  Paper Area F1      : {metrics['area_f1']:.4f}")
        print("=" * 50)

    elif args.command == "sweep":
        print(f"Running confidence threshold sweep on {args.split} split...")
        model = FullModel(cfg.model).to(device)
        load_checkpoint(model, path=args.checkpoint, device=device)
        _, val_loader, test_loader, _ = build_loaders(cfg)
        loader = val_loader if args.split == "val" else test_loader

        results = sweep_confidence_thresholds(model, loader, device)
        best = results["best"]
        print("=" * 65)
        print(f"{'Threshold':>10} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10}")
        print("-" * 65)
        for r in results["all"]:
            mark = " <-- (Optimal Peak)" if r["threshold"] == best["threshold"] else ""
            print(f"{r['threshold']:>10.2f} | {r['precision']:>10.4f} | {r['recall']:>10.4f} | {r['f1']:>10.4f}{mark}")
        print("=" * 65)
        print(f"Optimal Threshold: {best['threshold']:.2f}")
        print(f"  Peak Precision : {best['precision']:.4f} ({best['precision']*100:.2f}%)")
        print(f"  Peak Recall    : {best['recall']:.4f} ({best['recall']*100:.2f}%)")
        print(f"  Peak F1-Score  : {best['f1']:.4f}")
        print("=" * 65)


if __name__ == "__main__":
    main()
