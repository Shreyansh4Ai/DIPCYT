"""
DIPCYT-X40s — Production Training Pipeline (Hardware-Safe, Long-Run)
====================================================================
Engineered for unattended long-run execution on a single GPU (or CPU fallback):
  1. Startup VRAM probe: Binary-probes GPU capacity with 20% safety margin.
  2. Mixed Precision: torch.cuda.amp (float16/bfloat16) + GradScaler.
  3. Gradient Accumulation: Emulates target effective batch size with micro-batches.
  4. torch.cuda.OutOfMemoryError Guard: Catches OOM, clears cache, recovers gracefully.
  5. NaN/Inf Guard: Skips corrupt gradient updates and logs diagnostics.
  6. Atomic Checkpointing with Disk Cap: Retains best_model + last-k checkpoints only.
  7. Dual Console + File Logging: Persists timestamp, GPU VRAM %, loss, LR, and ETA.
  8. Proxy Overfit Runner: Reusable proxy budget (N=10, 800 iters) for Optuna HPO.
"""

from __future__ import annotations

import argparse
import copy
import datetime
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import Config, HardwareConfig, TrainConfig
from .dataset import build_loaders, polygon_collate_fn
from .eval import evaluate_model
from .losses import PolygonHungarianLoss
from .model import FullModel


# ── Persistent Disk & Console Logger ─────────────────────────────────────────

class PipelineLogger:
    """Logs messages simultaneously to stdout and a persistent file."""
    def __init__(self, log_path: Optional[str | Path] = None):
        self.log_path = Path(log_path) if log_path else None
        if self.log_path:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str, console: bool = True) -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {msg}"
        if console:
            try:
                print(formatted)
            except UnicodeEncodeError:
                safe_str = formatted.encode("ascii", errors="replace").decode("ascii")
                print(safe_str)
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(formatted + "\n")
            except Exception:
                pass


# ── VRAM Probe & Dynamic Batch Sizing ────────────────────────────────────────

def probe_vram_and_batch_size(
    model: nn.Module,
    cfg: Config,
    device: torch.device,
    logger: PipelineLogger,
) -> Tuple[int, int]:
    """
    Tests synthetic forward+backward passes on candidate micro-batch sizes
    [1, 2, 4, 8, 16, 32] on GPU. Backs off 20% to prevent mid-training OOMs.
    Returns (safe_micro_batch_size, gradient_accumulation_steps).
    """
    if device.type != "cuda":
        logger.log("  [VRAM Probe] Device is CPU. Using batch_size=2, accum_steps=4.")
        return 2, 4

    target_effective = cfg.hardware.target_effective_batch_size
    candidates = [1, 2, 4, 8, 16, 32]
    max_safe = 1

    logger.log("  [VRAM Probe] Probing single-GPU memory capacity...")
    torch.cuda.empty_cache()

    criterion = PolygonHungarianLoss(loss_cfg=cfg.loss, matcher_cfg=cfg.matcher)
    use_amp = cfg.hardware.use_amp
    amp_dtype = torch.bfloat16 if cfg.hardware.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported() else torch.float16

    for bs in candidates:
        try:
            torch.cuda.empty_cache()
            dummy_imgs = torch.randn(bs, 3, cfg.model.img_size, cfg.model.img_size, device=device)
            dummy_targets = []
            for _ in range(bs):
                polys = torch.rand(16, cfg.model.num_points, 2, device=device)
                vmask = torch.ones(16, dtype=torch.bool, device=device)
                pmask = torch.ones(16, cfg.model.num_points, dtype=torch.bool, device=device)
                lbls  = torch.ones(16, dtype=torch.long, device=device)
                dummy_targets.append({
                    "polygons": polys,
                    "valid_mask": vmask,
                    "point_mask": pmask,
                    "labels": lbls,
                })

            model.train()
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out = model(dummy_imgs)
                loss = criterion(out, dummy_targets)
            loss.backward()

            model.zero_grad(set_to_none=True)
            del dummy_imgs, dummy_targets, out, loss
            torch.cuda.empty_cache()

            max_safe = bs
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            logger.log(f"  [VRAM Probe] Micro-batch={bs} failed (OOM).")
            torch.cuda.empty_cache()
            break

    # Apply 20% safety margin backoff
    margin = cfg.hardware.vram_safety_margin
    backed_off = max(1, int(max_safe * (1.0 - margin)))
    # Round to power-of-2 or integer
    final_micro_bs = backed_off
    accum_steps = max(1, target_effective // final_micro_bs)

    vram_alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
    vram_total = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)

    logger.log(f"  [OK] [VRAM Probe Complete] Max feasible micro-batch: {max_safe} | "
               f"Safe micro-batch (20% margin): {final_micro_bs} | "
               f"Grad Accumulation: {accum_steps} | "
               f"Effective Batch: {final_micro_bs * accum_steps} | "
               f"GPU VRAM: {vram_alloc:.1f} MB / {vram_total:.1f} MB")

    return final_micro_bs, accum_steps


# ── Optimizer & Scheduler Builder ────────────────────────────────────────────

def build_optimizer_and_scheduler(
    model: nn.Module,
    cfg: Config,
    total_steps: int,
) -> Tuple[torch.optim.Optimizer, Optional[Any]]:
    tcfg = cfg.train
    params = [p for p in model.parameters() if p.requires_grad]

    if tcfg.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=tcfg.lr,
            momentum=tcfg.sgd_momentum,
            weight_decay=tcfg.weight_decay,
            nesterov=True,
        )
    else:  # default AdamW
        optimizer = torch.optim.AdamW(
            params,
            lr=tcfg.lr,
            weight_decay=tcfg.weight_decay,
            betas=(0.9, 0.999),
        )

    scheduler = None
    if tcfg.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(total_steps, 1),
            eta_min=tcfg.lr * 1e-3,
        )
    elif tcfg.scheduler == "linear_warmup_decay":
        warmup_steps = tcfg.warmup_steps
        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(1e-3, 1.0 - progress)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


# ── Checkpointing & Disk Usage Cap ───────────────────────────────────────────

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    step: int,
    metric: float,
    path: str | Path,
    keep_last_k: int = 3,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")

    ckpt = {
        "epoch": epoch,
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "metric": metric,
    }
    torch.save(ckpt, tmp_path)
    if path.exists():
        bak_path = path.with_suffix(".bak")
        if bak_path.exists():
            bak_path.unlink()
        path.rename(bak_path)
    tmp_path.rename(path)

    # Clean up older numbered epoch checkpoints to cap disk usage
    if keep_last_k > 0 and path.parent.exists():
        epoch_ckpts = sorted(
            [p for p in path.parent.glob("epoch_*.pth") if p.name != "best_model.pth"],
            key=lambda p: p.stat().st_mtime,
        )
        while len(epoch_ckpts) > keep_last_k:
            oldest = epoch_ckpts.pop(0)
            try:
                oldest.unlink()
            except Exception:
                pass


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    path: str | Path = "checkpoint_best.pth",
    device: torch.device = torch.device("cpu"),
) -> Tuple[int, int, float]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {path}")

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None and ckpt.get("optimizer_state_dict"):
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception:
            pass
    if scheduler is not None and ckpt.get("scheduler_state_dict"):
        try:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        except Exception:
            pass
    if scaler is not None and ckpt.get("scaler_state_dict"):
        try:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        except Exception:
            pass
    model.to(device)
    return ckpt.get("epoch", 0), ckpt.get("step", 0), ckpt.get("metric", 0.0)


# ── Single Epoch Training with OOM & NaN Guards ──────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[torch.cuda.amp.GradScaler],
    device: torch.device,
    cfg: Config,
    logger: PipelineLogger,
    epoch_idx: int = 0,
    accum_steps: int = 1,
) -> float:
    model.train()
    total_loss = 0.0
    valid_batches = 0
    nan_batches = 0
    oom_events = 0

    use_amp = cfg.hardware.use_amp and (device.type == "cuda")
    amp_dtype = (
        torch.bfloat16
        if cfg.hardware.amp_dtype == "bfloat16" and torch.cuda.is_bf16_supported()
        else torch.float16
    )

    optimizer.zero_grad(set_to_none=True)
    pbar = tqdm(loader, desc=f"Epoch {epoch_idx+1:02d} [Train]", leave=False)
    t0 = time.time()

    for batch_idx, (images, targets) in enumerate(pbar):
        # ── Step wrapped in OOM-safe block ──
        for retry in range(cfg.hardware.max_oom_retries):
            try:
                images = images.to(device, dtype=torch.float32)
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    scaled_loss = loss / accum_steps

                loss_val = float(loss.detach())
                if math.isnan(loss_val) or math.isinf(loss_val):
                    nan_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    logger.log(f"  [WARNING] [NaN/Inf Guard] Corrupted loss={loss_val} at batch {batch_idx}. Skipping update.")
                    break

                if scaler is not None and use_amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                # Step optimizer when accumulation threshold is reached
                if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
                    if cfg.train.grad_clip > 0:
                        if scaler is not None and use_amp:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.train.grad_clip)

                    if scaler is not None and use_amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                    if scheduler is not None:
                        scheduler.step()

                    optimizer.zero_grad(set_to_none=True)

                total_loss += loss_val
                valid_batches += 1
                pbar.set_postfix({"loss": f"{loss_val:.4f}"})

                # ── Periodic Step Logging ──
                if (batch_idx + 1) % cfg.hardware.log_every_n_steps == 0:
                    lr_curr = optimizer.param_groups[0]["lr"]
                    vram_str = ""
                    if device.type == "cuda":
                        alloc = torch.cuda.memory_allocated(device) / (1024 ** 2)
                        total_mem = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
                        pct = 100.0 * alloc / total_mem
                        vram_str = f" | VRAM: {alloc:.0f}MB/{total_mem:.0f}MB ({pct:.1f}%)"
                    logger.log(
                        f"Epoch {epoch_idx+1:02d} [{batch_idx+1:04d}/{len(loader):04d}] "
                        f"Loss: {loss_val:.4f} | LR: {lr_curr:.2e}{vram_str}",
                        console=False,
                    )
                break  # Successful pass, break retry loop

            except (torch.cuda.OutOfMemoryError, RuntimeError, Exception) as e:
                oom_events += 1
                logger.log(f"  [WARNING] [OOM Guard] Caught {type(e).__name__} at batch {batch_idx}. Clearing cache (Retry {retry+1}).")
                images = None
                outputs = None
                loss = None
                scaled_loss = None
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
                try:
                    optimizer.zero_grad(set_to_none=True)
                except Exception:
                    pass
                if retry == cfg.hardware.max_oom_retries - 1:
                    logger.log(f"  [ERROR] [OOM Guard] Batch {batch_idx} dropped after {cfg.hardware.max_oom_retries} retries.")
                    break

    if oom_events > 0:
        logger.log(f"  [WARNING] Total OOM events recovered in epoch {epoch_idx+1}: {oom_events}")
    if nan_batches > 0:
        logger.log(f"  [WARNING] Total NaN/Inf batches skipped in epoch {epoch_idx+1}: {nan_batches}")

    return total_loss / max(1, valid_batches)


# -- Full Training Pipeline (Production Runner) -------------------------------

def train_pipeline(
    cfg: Config,
    resume_path: Optional[str | Path] = None,
) -> FullModel:
    """Production training pipeline with baseline matching and safety guards."""
    logger = PipelineLogger(cfg.hardware.log_file)

    # 1. Deterministic Seeding
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)

    # 2. Device Selection & Warnings
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.log(f"[OK] Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logger.log("[WARNING] CUDA is NOT available. Running on CPU (slowdown expected).", console=True)

    # 3. Build Model & Gradient Checkpointing
    model = FullModel(cfg.model).to(device)
    if cfg.model.enable_gradient_checkpointing:
        model.enable_gradient_checkpointing(True)

    criterion = PolygonHungarianLoss(loss_cfg=cfg.loss, matcher_cfg=cfg.matcher)

    # 4. VRAM Probe & Dynamic Batch Configuration
    if cfg.hardware.auto_probe_vram and device.type == "cuda":
        micro_bs, accum_steps = probe_vram_and_batch_size(model, cfg, device, logger)
        cfg.hardware.micro_batch_size = micro_bs
        cfg.hardware.gradient_accumulation_steps = accum_steps
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    else:
        micro_bs = cfg.hardware.micro_batch_size
        accum_steps = cfg.hardware.gradient_accumulation_steps

    # 5. Build Dataloaders
    train_loader, val_loader, test_loader, _ = build_loaders(cfg)

    # 6. Resume Checkpoint / Model Weights if specified
    start_epoch = 0
    best_f1 = -1.0
    ckpt_dict = None
    if resume_path:
        path = Path(resume_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {path}")
        ckpt_dict = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_dict["model_state_dict"])
        saved_epoch = ckpt_dict.get("epoch", 0)
        best_f1 = ckpt_dict.get("metric", -1.0)
        start_epoch = saved_epoch + 1
        model.to(device)

    # 7. Backbone Freeze / Unfreeze based on start_epoch
    if cfg.train.backbone_unfreeze_epoch > start_epoch:
        model.freeze_backbone()
        logger.log(f"Backbone frozen until epoch {cfg.train.backbone_unfreeze_epoch}")
    else:
        model.unfreeze_backbone()
        logger.log("Backbone is unfrozen (end-to-end training active)")

    # 8. Build Optimizer, Scheduler, and GradScaler with correct parameter groups
    remaining_steps = (len(train_loader) // accum_steps) * max(1, cfg.train.num_epochs - start_epoch)
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, remaining_steps)
    use_amp = cfg.hardware.use_amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    if ckpt_dict is not None:
        if not getattr(cfg.train, "reset_optimizer", False) and ckpt_dict.get("optimizer_state_dict"):
            try:
                optimizer.load_state_dict(ckpt_dict["optimizer_state_dict"])
            except Exception:
                pass
        else:
            for pg in optimizer.param_groups:
                scale = pg.get("lr_scale", 1.0)
                pg["lr"] = cfg.train.lr * scale
                pg["initial_lr"] = cfg.train.lr * scale

        if not getattr(cfg.train, "reset_optimizer", False) and ckpt_dict.get("scheduler_state_dict") and scheduler is not None:
            try:
                scheduler.load_state_dict(ckpt_dict["scheduler_state_dict"])
            except Exception:
                pass

        if ckpt_dict.get("scaler_state_dict") and scaler is not None:
            try:
                scaler.load_state_dict(ckpt_dict["scaler_state_dict"])
            except Exception:
                pass
        logger.log(f"[OK] Resumed training from {resume_path} (resuming at Epoch {start_epoch+1:02d}, best F1: {best_f1:.4f})")

    ckpt_dir = Path(cfg.train.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger.log("=" * 60)
    logger.log(f"Starting Production Training: {cfg.train.num_epochs} Epochs")
    logger.log(f"  Optimizer: {cfg.train.optimizer} | LR: {cfg.train.lr:.2e} | Scheduler: {cfg.train.scheduler}")
    logger.log(f"  Micro-Batch: {micro_bs} | Accumulation: {accum_steps} | Effective Batch: {micro_bs * accum_steps}")
    logger.log(f"  AMP Enabled: {use_amp} | Workers: {cfg.hardware.num_workers}")
    logger.log(f"  Training Samples: {len(train_loader.dataset):,} | Validation Samples: {len(val_loader.dataset):,}")
    logger.log("=" * 60)

    # 9. Main Epoch Loop
    for epoch in range(start_epoch, cfg.train.num_epochs):
        # Check unfreeze
        if cfg.train.backbone_unfreeze_epoch > 0 and epoch == cfg.train.backbone_unfreeze_epoch:
            model.unfreeze_backbone()
            remaining_steps = (len(train_loader) // accum_steps) * (cfg.train.num_epochs - epoch)
            optimizer, scheduler = build_optimizer_and_scheduler(model, cfg, remaining_steps)
            logger.log(f"[OK] Epoch {epoch+1:02d}: Backbone unfrozen, optimizer rebuilt.")

        epoch_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            cfg=cfg,
            logger=logger,
            epoch_idx=epoch,
            accum_steps=accum_steps,
        )

        # Validation
        val_metrics = evaluate_model(model, val_loader, device)
        f1 = val_metrics["f1"]
        area_f1 = val_metrics["area_f1"]

        log_msg = (
            f"Epoch {epoch+1:02d}/{cfg.train.num_epochs:02d} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Val P: {val_metrics['precision']:.4f} R: {val_metrics['recall']:.4f} F1: {f1:.4f} | "
            f"Paper Area F1: {area_f1:.4f}"
        )
        logger.log(log_msg)

        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=(epoch + 1) * len(train_loader),
                metric=f1,
                path=ckpt_dir / "best_model.pth",
                keep_last_k=cfg.hardware.keep_last_k_checkpoints,
            )
            logger.log(f"  [BEST] New best model saved! (Val F1: {best_f1:.4f})")

        # Save regular epoch checkpoint
        if (epoch + 1) % cfg.train.save_every_epoch == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                step=(epoch + 1) * len(train_loader),
                metric=f1,
                path=ckpt_dir / f"epoch_{epoch+1:03d}.pth",
                keep_last_k=cfg.hardware.keep_last_k_checkpoints,
            )

    logger.log("=" * 65)
    logger.log(f"Training Complete! Peak Validation F1: {best_f1:.4f}")
    logger.log("=" * 65)
    return model


# ── Proxy Trial Runner (Progressive Overfit for Optuna HPO) ──────────────────

def run_proxy_trial(
    cfg: Config,
    device: Optional[torch.device] = None,
    trial: Optional[Any] = None,
) -> float:
    """
    Fast trial proxy: overfit on N=10 fixed images for 800 iterations.
    Employs AMP, OOM recovery, and Optuna intermediate report/pruning.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Strict reproducibility per trial
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    random.seed(cfg.train.seed)

    model = FullModel(cfg.model).to(device)
    criterion = PolygonHungarianLoss(loss_cfg=cfg.loss, matcher_cfg=cfg.matcher)

    # Data loading for proxy partition
    from .dataset import COCOTextAnnotationStore, COCOTextDataset
    ann_store = COCOTextAnnotationStore(cfg.data.ann_file)
    train_ids, val_ids, _ = ann_store.get_splits(seed=cfg.train.seed)

    proxy_ids = train_ids[:cfg.train.proxy_num_images]
    val_proxy_ids = val_ids[:max(cfg.train.proxy_num_images, 10)]

    proxy_train_dataset = COCOTextDataset(
        img_ids=proxy_ids,
        ann_store=ann_store,
        img_dir=cfg.data.img_dir,
        img_size=cfg.model.img_size,
        num_points=cfg.model.num_points,
        max_targets=cfg.data.max_targets,
        augment=False,
        min_poly_area=cfg.data.min_poly_area,
        min_box_pixels=cfg.data.min_box_pixels,
    )
    val_proxy_dataset = COCOTextDataset(
        img_ids=val_proxy_ids,
        ann_store=ann_store,
        img_dir=cfg.data.img_dir,
        img_size=cfg.model.img_size,
        num_points=cfg.model.num_points,
        max_targets=cfg.data.max_targets,
        augment=False,
        min_poly_area=cfg.data.min_poly_area,
        min_box_pixels=cfg.data.min_box_pixels,
    )

    bs = min(cfg.hardware.micro_batch_size, len(proxy_ids))
    proxy_loader = DataLoader(
        proxy_train_dataset,
        batch_size=bs,
        shuffle=True,
        collate_fn=polygon_collate_fn,
    )
    val_loader = DataLoader(
        val_proxy_dataset,
        batch_size=bs,
        shuffle=False,
        collate_fn=polygon_collate_fn,
    )

    optimizer, scheduler = build_optimizer_and_scheduler(
        model, cfg, total_steps=cfg.train.proxy_num_iters
    )

    use_amp = cfg.hardware.use_amp and (device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None

    num_iters = cfg.train.proxy_num_iters
    log_every = cfg.train.proxy_log_every
    iter_loader = iter(proxy_loader)

    for it in range(1, num_iters + 1):
        model.train()
        try:
            images, targets = next(iter_loader)
        except StopIteration:
            iter_loader = iter(proxy_loader)
            images, targets = next(iter_loader)

        try:
            images = images.to(device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, targets)

            loss_val = float(loss.detach())
            if math.isnan(loss_val) or math.isinf(loss_val):
                return 0.0

            if scaler is not None and use_amp:
                scaler.scale(loss).backward()
                if cfg.train.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.train.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()

            if scheduler is not None:
                scheduler.step()

        except (torch.cuda.OutOfMemoryError, RuntimeError):
            torch.cuda.empty_cache()
            return 0.0

        # Periodic intermediate evaluation and pruning check
        if it % log_every == 0 or it == num_iters:
            val_metrics = evaluate_model(model, val_loader, device)
            f1 = val_metrics["f1"]
            if trial is not None:
                trial.report(f1, step=it)
                if trial.should_prune():
                    import optuna
                    raise optuna.TrialPruned()

    final_metrics = evaluate_model(model, val_loader, device)
    return float(final_metrics["f1"])


# ── CLI Entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DIPCYT-X40s Production Training")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config) if args.config else Config()
    train_pipeline(cfg, resume_path=args.resume)
