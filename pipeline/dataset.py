"""
DIPCYT-X40s — Dataset & Baseline-Matched Preprocessing
======================================================
Implements:
  1. Baseline-Matched Image Preprocessing (Roy et al., Pattern Recognition 2024):
     - Input resolution: 640 × 640 [PDF Sec 4.2: 'All images are reshaped to 640 × 640']
     - Interpolation: Bilinear (cv2.INTER_LINEAR)
     - Normalization: ImageNet standard mean [0.485, 0.456, 0.406] and std [0.229, 0.224, 0.225]
     - 6-Point Contour Resampling [PDF Sec 3.3: 'six nodes to support polygonal predictions']
     - Polygon Canonicalisation: Clockwise winding + top-left-most vertex start
     - Minimum Bounding Box Filter: 26px threshold to eliminate unresolvable stride-40 noise.
  2. Augmentation Policy:
     - [ASSUMPTION]: The PDF specifies model configurations follow TPH-YOLOv5++ [11],
       pre-training on COCO and fine-tuning. For exact parameters not listed in the PDF,
       we default to standard COCO-Text / YOLO augmentations (scale crop 0.7-1.3,
       horizontal flip p=0.5, rotation ±10°, HSV color jitter, Gaussian blur).
  3. Single-GPU Laptop Dataloader Safety:
     - CPU core-aware worker count (capped at 4 to prevent thrashing)
     - persistent_workers=True to avoid sub-process respawn stalls
     - Deterministic worker seeding + PyTorch Generator.
"""

from __future__ import annotations

import cv2
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .config import Config, DataConfig, HardwareConfig, TrainConfig


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def _canonicalize_polygon(poly: np.ndarray) -> np.ndarray:
    """
    Forces consistent clockwise winding and a fixed start vertex (top-left-most point).
    Removes index-order ambiguity that destabilises Hungarian matching and L1 loss.
    Input/Output: (N, 2) float32 in [0, 1] normalized coordinates.
    """
    poly = np.asarray(poly, dtype=np.float32)
    n = len(poly)
    if n < 3:
        return poly.copy()
    x, y = poly[:, 0], poly[:, 1]
    signed_area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if signed_area < 0:  # Force clockwise
        poly = poly[::-1].copy()
    start = int(np.argmin(poly[:, 0] + poly[:, 1]))
    poly = np.roll(poly, -start, axis=0)
    return poly.astype(np.float32)


def _resample_polygon(pts: np.ndarray, n: int) -> np.ndarray:
    """
    Uniformly resamples a polygon contour along its perimeter arc-length to n vertices.
    [PDF Sec 3.3]: 'employs six nodes to support polygonal predictions'
    """
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == n:
        return pts.copy()
    closed = np.vstack([pts, pts[:1]])
    seg = np.sqrt(((closed[1:] - closed[:-1]) ** 2).sum(1))
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total <= 1e-8:
        return np.repeat(pts[:1], n, axis=0).astype(np.float32)
    targets = np.linspace(0.0, total, n, endpoint=False)
    out = []
    for t in targets:
        k = int(np.searchsorted(cum, t, side="right") - 1)
        k = min(max(k, 0), len(seg) - 1)
        denom = seg[k] if seg[k] > 1e-8 else 1.0
        r = (t - cum[k]) / denom
        out.append(closed[k] + r * (closed[k + 1] - closed[k]))
    return np.asarray(out, dtype=np.float32)


def pad_polygon(poly: np.ndarray, max_points: int) -> Tuple[np.ndarray, int]:
    """Pads polygon array to max_points with zeros for batch collation."""
    n = len(poly)
    real_count = min(n, max_points)
    if n >= max_points:
        return poly[:max_points], real_count
    pad = np.zeros((max_points - n, 2), dtype=np.float32)
    return np.concatenate([poly, pad], axis=0), real_count


def polygon_area(poly: np.ndarray) -> float:
    """Computes Shoelace polygon area."""
    n = len(poly)
    if n < 3:
        return 0.0
    x = poly[:, 0]
    y = poly[:, 1]
    return 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])
                     + x[-1] * y[0] - x[0] * y[-1])


def random_horizontal_flip(image: np.ndarray, polys: List[np.ndarray]) -> Tuple[np.ndarray, List[np.ndarray]]:
    image = cv2.flip(image, 1)
    for p in polys:
        p[:, 0] = 1.0 - p[:, 0]
    return image, polys


def random_scale_crop(
    image: np.ndarray,
    polys: List[np.ndarray],
    scale_range: Tuple[float, float] = (0.7, 1.3),
) -> Tuple[np.ndarray, List[np.ndarray]]:
    h, w = image.shape[:2]
    scale = random.uniform(*scale_range)
    new_w = int(w * scale)
    new_h = int(h * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if scale > 1.0:
        x0 = random.randint(0, new_w - w)
        y0 = random.randint(0, new_h - h)
        image = image[y0:y0 + h, x0:x0 + w]
        for p in polys:
            p[:, 0] = (p[:, 0] * new_w - x0) / w
            p[:, 1] = (p[:, 1] * new_h - y0) / h
    else:
        pad_img = np.zeros((h, w, 3), dtype=image.dtype)
        pad_img[:new_h, :new_w] = image
        image = pad_img
        for p in polys:
            p[:, 0] = p[:, 0] * new_w / w
            p[:, 1] = p[:, 1] * new_h / h
    for p in polys:
        p[:, 0] = np.clip(p[:, 0], 0.0, 1.0)
        p[:, 1] = np.clip(p[:, 1], 0.0, 1.0)
    return image, polys


def random_color_jitter(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 0] += random.uniform(-10, 10)
    hsv[:, :, 1] *= random.uniform(0.7, 1.3)
    hsv[:, :, 2] *= random.uniform(0.6, 1.4)
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def random_rotate(
    image: np.ndarray,
    polys: List[np.ndarray],
    max_angle: int = 10,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    h, w = image.shape[:2]
    angle = random.uniform(-max_angle, max_angle)
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    image = cv2.warpAffine(image, M, (w, h),
                           flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REFLECT_101)
    for p in polys:
        pts = p.copy()
        pts[:, 0] *= w
        pts[:, 1] *= h
        ones = np.ones((len(pts), 1))
        pts_h = np.hstack([pts, ones])
        pts_r = (M @ pts_h.T).T
        pts_r[:, 0] = np.clip(pts_r[:, 0] / w, 0.0, 1.0)
        pts_r[:, 1] = np.clip(pts_r[:, 1] / h, 0.0, 1.0)
        p[:] = pts_r
    return image, polys


def random_gaussian_blur(image: np.ndarray) -> np.ndarray:
    k = random.choice([3, 5])
    return cv2.GaussianBlur(image, (k, k), 0)


class COCOTextAnnotationStore:
    """Loads, filters (legible-only), and partitions COCO-Text v2.0 annotations using compact caching."""
    def __init__(self, ann_file: str | Path):
        import gc
        import pickle

        ann_file = Path(ann_file)
        cache_file = ann_file.parent / "cocotext_compact.pkl" if ann_file.parent.exists() else Path("cocotext_compact.pkl")

        if cache_file.exists():
            print(f"  [COCOText] Fast loading compact annotation cache: {cache_file} ...")
            with open(cache_file, "rb") as f:
                data = pickle.load(f)
            self.imgs = data["imgs"]
            self.img_to_anns = data["anns"]
            self.legible_img_ids = list(self.img_to_anns.keys())
            print(f"  [COCOText] Loaded {len(self.legible_img_ids):,} images with legible annotations.")
            return

        if not ann_file.exists():
            print(f"  [COCOText] Warning: Annotation file {ann_file} not found. Creating placeholder index.")
            self.imgs = {}
            self.legible_img_ids = []
            self.img_to_anns = {}
            return

        print(f"  [COCOText] Parsing raw {ann_file} and generating compact cache ...")
        with open(ann_file, "r", encoding="utf-8") as f:
            ct = json.load(f)

        self.legible_img_ids: List[str] = []
        self.img_to_anns: Dict[str, List[dict]] = {}
        for ann_id, ann in ct.get("anns", {}).items():
            if ann.get("legibility") == "legible":
                img_id = str(ann["image_id"])
                if img_id not in self.img_to_anns:
                    self.legible_img_ids.append(img_id)
                self.img_to_anns.setdefault(img_id, []).append({
                    "pts": ann.get("polygon", []),
                    "bbox": ann.get("bbox", []),
                    "mask": ann.get("mask", None),
                })

        self.imgs = {}
        for img_id in self.img_to_anns:
            img_info = ct.get("imgs", {}).get(img_id, {})
            self.imgs[img_id] = {
                "file_name": img_info.get("file_name", f"COCO_train2014_{int(img_id):012d}.jpg"),
                "width": img_info.get("width"),
                "height": img_info.get("height"),
                "set": img_info.get("set", "train"),
            }

        del ct
        gc.collect()

        try:
            with open(cache_file, "wb") as f:
                pickle.dump({"imgs": self.imgs, "anns": self.img_to_anns}, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"  [COCOText] Cached compact annotations to {cache_file} ({cache_file.stat().st_size / (1024*1024):.2f} MB).")
        except Exception as e:
            print(f"  [COCOText] Note: cache write skipped: {e}")

    def get_splits(self, seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
        """
        Splits dataset into train, val, test.
        [PDF Sec 4.2]: 'A 70-30-20% training-validation-testing split is used for each dataset.'
        Carves test set from COCO-Text's val partition.
        """
        train_ids, val_ids = [], []
        for img_id in self.legible_img_ids:
            split = self.imgs.get(img_id, {}).get("set", "train")
            if split == "train":
                train_ids.append(img_id)
            else:
                val_ids.append(img_id)

        rng = random.Random(seed)
        val_ids_copy = list(val_ids)
        rng.shuffle(val_ids_copy)
        split_idx = int(0.67 * len(val_ids_copy))
        val_final = val_ids_copy[:split_idx]
        test_final = val_ids_copy[split_idx:]
        return train_ids, val_final, test_final

    def get_polygons_for_image(
        self,
        img_id: str,
        num_points: int = 6,
        img_dir: Optional[Path] = None,
    ) -> Tuple[List[np.ndarray], str, int, int]:
        anns = self.img_to_anns.get(img_id, [])
        img_info = self.imgs.get(img_id, {})
        fname = img_info.get("file_name", f"COCO_train2014_{int(img_id):012d}.jpg")
        img_w = img_info.get("width", None)
        img_h = img_info.get("height", None)

        if img_w is None or img_h is None:
            if img_dir is not None:
                p = img_dir / fname
                im = cv2.imread(str(p))
                if im is not None:
                    img_h, img_w = im.shape[:2]
                else:
                    img_h, img_w = 640, 640
            else:
                img_h, img_w = 640, 640

        polys = []
        for ann in anns:
            poly = None
            mask = ann.get("mask", None) or ann.get("polygon", None) or ann.get("pts", None)
            if mask is not None and len(mask) >= 6:
                try:
                    arr = np.asarray(mask, dtype=np.float32).reshape(-1, 2)
                    if len(arr) >= 3:
                        poly = _resample_polygon(arr, num_points)
                except Exception:
                    poly = None

            if poly is None:
                bbox = ann.get("bbox", None)
                if bbox is None:
                    continue
                x, y, bw, bh = bbox
                poly = np.array([[x, y], [x + bw, y],
                                 [x + bw, y + bh], [x, y + bh]], dtype=np.float32)
                if num_points != 4:
                    poly = _resample_polygon(poly, num_points)

            poly = poly.astype(np.float32).copy()
            poly[:, 0] /= float(img_w)
            poly[:, 1] /= float(img_h)
            poly = np.clip(poly, 0.0, 1.0)
            polys.append(poly)

        return polys, fname, int(img_h), int(img_w)


class COCOTextDataset(Dataset):
    """
    COCO-Text v2.0 text localization dataset with baseline-matched preprocessing.
    """
    def __init__(
        self,
        img_ids: List[str],
        ann_store: COCOTextAnnotationStore,
        img_dir: str | Path,
        img_size: int = 640,
        num_points: int = 6,
        max_targets: int = 128,
        augment: bool = False,
        min_poly_area: float = 1e-6,
        min_box_pixels: int = 26,
        aug_cfg: Optional[DataConfig] = None,
    ):
        self.img_ids = img_ids
        self.ann_store = ann_store
        self.img_dir = Path(img_dir)
        self.img_size = img_size
        self.num_points = num_points
        self.max_targets = max_targets
        self.augment = augment
        self.min_poly_area = min_poly_area
        self.min_box_pixels = min_box_pixels
        self.aug_cfg = aug_cfg or DataConfig()

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        img_id = self.img_ids[idx]
        normed, fname, h, w = self.ann_store.get_polygons_for_image(
            img_id, num_points=self.num_points, img_dir=self.img_dir
        )

        img_path = self.img_dir / fname
        image = cv2.imread(str(img_path))
        if image is None:
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            h, w = self.img_size, self.img_size
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w = image.shape[:2]

        if self.augment and len(normed) > 0:
            if random.random() < self.aug_cfg.flip_prob:
                image, normed = random_horizontal_flip(image, normed)
            image, normed = random_scale_crop(
                image, normed,
                scale_range=(self.aug_cfg.scale_range_lo, self.aug_cfg.scale_range_hi)
            )
            if random.random() < self.aug_cfg.rotate_prob:
                image, normed = random_rotate(image, normed, max_angle=self.aug_cfg.rotate_max_angle)
            if random.random() < self.aug_cfg.jitter_prob:
                image = random_color_jitter(image)
            if random.random() < self.aug_cfg.blur_prob:
                image = random_gaussian_blur(image)

        normed = [p for p in normed if polygon_area(p) > self.min_poly_area]
        normed = [
            p for p in normed
            if max(p[:, 0].max() - p[:, 0].min(),
                   p[:, 1].max() - p[:, 1].min()) * self.img_size >= self.min_box_pixels
        ]

        normed = [_canonicalize_polygon(p) for p in normed]

        image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        for p in normed:
            pts = (p * self.img_size).astype(np.int32)
            cv2.fillPoly(mask, [pts], color=1)


        padded, real_counts = [], []
        for p in normed:
            pp, rc = pad_polygon(p, self.num_points)
            padded.append(pp)
            real_counts.append(rc)

        num_real = min(len(padded), self.max_targets)
        polys_np = np.zeros((self.max_targets, self.num_points, 2), dtype=np.float32)
        for i, p in enumerate(padded[:self.max_targets]):
            polys_np[i] = p

        polys_tensor = torch.from_numpy(polys_np.copy())
        valid_mask = torch.zeros(self.max_targets, dtype=torch.bool)
        valid_mask[:num_real] = True
        labels = torch.zeros(self.max_targets, dtype=torch.long)
        labels[:num_real] = 1

        point_mask = torch.zeros(self.max_targets, self.num_points, dtype=torch.bool)
        for i in range(num_real):
            point_mask[i, :real_counts[i]] = True


        image_tensor = (
            torch.from_numpy(image.copy())
            .permute(2, 0, 1).float().div(255.0)
            .sub(IMAGENET_MEAN).div(IMAGENET_STD)
        )
        mask_tensor = torch.from_numpy(mask.copy()).unsqueeze(0).float()

        return image_tensor, {
            "polygons": polys_tensor,
            "labels": labels,
            "valid_mask": valid_mask,
            "point_mask": point_mask,
            "orig_size": torch.tensor([h, w]),
            "mask": mask_tensor,
        }


def polygon_collate_fn(batch: List[Tuple[torch.Tensor, dict]]) -> Tuple[torch.Tensor, List[dict]]:
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


def _seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(
    cfg: Config,
    ann_store: Optional[COCOTextAnnotationStore] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, COCOTextAnnotationStore]:
    """
    Builds single-GPU, laptop-safe (train_loader, val_loader, test_loader, ann_store).
    Auto-tunes worker counts to CPU capabilities.
    """
    if ann_store is None:
        ann_store = COCOTextAnnotationStore(cfg.data.ann_file)

    train_ids, val_ids, test_ids = ann_store.get_splits(seed=cfg.train.seed)

    train_dataset = COCOTextDataset(
        img_ids=train_ids,
        ann_store=ann_store,
        img_dir=cfg.data.img_dir,
        img_size=cfg.model.img_size,
        num_points=cfg.model.num_points,
        max_targets=cfg.data.max_targets,
        augment=cfg.data.augment,
        min_poly_area=cfg.data.min_poly_area,
        min_box_pixels=cfg.data.min_box_pixels,
        aug_cfg=cfg.data,
    )
    val_dataset = COCOTextDataset(
        img_ids=val_ids,
        ann_store=ann_store,
        img_dir=cfg.data.img_dir,
        img_size=cfg.model.img_size,
        num_points=cfg.model.num_points,
        max_targets=cfg.data.max_targets,
        augment=False,
        min_poly_area=cfg.data.min_poly_area,
        min_box_pixels=cfg.data.min_box_pixels,
    )
    test_dataset = COCOTextDataset(
        img_ids=test_ids,
        ann_store=ann_store,
        img_dir=cfg.data.img_dir,
        img_size=cfg.model.img_size,
        num_points=cfg.model.num_points,
        max_targets=cfg.data.max_targets,
        augment=False,
        min_poly_area=cfg.data.min_poly_area,
        min_box_pixels=cfg.data.min_box_pixels,
    )


    cpu_cores = os.cpu_count() or 1
    num_workers = min(cfg.hardware.num_workers, max(1, cpu_cores // 2), 4)
    use_pin_memory = cfg.hardware.pin_memory and torch.cuda.is_available()
    persistent = cfg.hardware.persistent_workers and (num_workers > 0)


    bs = getattr(cfg.hardware, "micro_batch_size", 4)

    g = torch.Generator()
    g.manual_seed(cfg.train.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        collate_fn=polygon_collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent,
        drop_last=True,
        worker_init_fn=_seed_worker,
        generator=g,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        collate_fn=polygon_collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent,
        drop_last=False,
        worker_init_fn=_seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=bs,
        shuffle=False,
        collate_fn=polygon_collate_fn,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=persistent,
        drop_last=False,
        worker_init_fn=_seed_worker,
    )

    return train_loader, val_loader, test_loader, ann_store
