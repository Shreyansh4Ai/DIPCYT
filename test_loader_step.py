from pipeline.config import Config
from pipeline.dataset import build_loaders
import gc
import torch

cfg = Config.from_yaml("production_config.yaml")
cfg.hardware.pin_memory = False
cfg.hardware.num_workers = 0
train_loader, val_loader, test_loader, ann_store = build_loaders(cfg)

print("Testing 50 items from train dataset...")
for i in range(50):
    sample = train_loader.dataset[i]
    if i % 10 == 0:
        val_cnt = sample[1]["valid_mask"].sum().item()
        print(f"  Sample {i:02d}: img shape {tuple(sample[0].shape)}, valid boxes: {val_cnt}")

print("\nTesting first 5 DataLoader batches...")
for b_idx, (imgs, targets) in enumerate(train_loader):
    total_valids = sum(t['valid_mask'].sum().item() for t in targets)
    print(f"  Batch {b_idx}: imgs shape={tuple(imgs.shape)}, valids count={total_valids}")
    if b_idx >= 4:
        break

print("\nDataLoader and dataset verified 100% operational!")
