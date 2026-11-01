"""
Progressive Overfit Verification Runner
========================================
Runs N=10 images overfit test for 800 iterations on CUDA.
Verifies model convergence, loss reduction, and polygon matching on real COCO-Text images.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
from pathlib import Path
import torch

from pipeline.config import Config
from pipeline.train import run_proxy_trial

if __name__ == "__main__":
    print("=" * 65)
    print("STEP 1: Starting Progressive Overfit Smoke Test (N=10 images, 800 iters)")
    print("=" * 65)

    cfg = Config.from_yaml("default_config.yaml") if Path("default_config.yaml").exists() else Config()
    cfg.train.proxy_num_images = 10
    cfg.train.proxy_num_iters = 800
    cfg.train.proxy_log_every = 100

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    final_f1 = run_proxy_trial(cfg=cfg, device=device)

    print("=" * 65)
    print(f"✓ Step 1 Complete! Progressive Overfit Final F1: {final_f1:.4f}")
    print("=" * 65)
