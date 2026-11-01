"""
Optuna Automated Hyperparameter Search Runner (Step 2)
======================================================
Runs 30 HPO trials with TPE Sampler & Hyperband Pruner.
Searches LR, loss weights, matcher costs, batch size, and attention reductions.
Exports best_params.yaml + study_trials.csv to optuna_results/.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
from pipeline.config import Config
from pipeline.optuna_search import run_optuna_search

if __name__ == "__main__":
    print("=" * 65)
    print("STEP 2: Launching Optuna Automated HPO Study (30 Trials)")
    print("Sampler: Tree-structured Parzen Estimator (TPE, Bergstra et al., 2011)")
    print("Pruner : Hyperband (Li et al., 2018)")
    print("=" * 65)

    cfg = Config.from_yaml("default_config.yaml") if Path("default_config.yaml").exists() else Config()

    study = run_optuna_search(
        base_cfg=cfg,
        n_trials=30,
        pruner_type="hyperband",
        output_dir="optuna_results",
    )

    print("=" * 65)
    print(f"✓ Step 2 Complete! Best Proxy Score: {study.best_value:.4f}")
    print(f"Winning parameters saved to: optuna_results/best_params.yaml")
    print("=" * 65)
