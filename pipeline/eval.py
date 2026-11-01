"""
DIPCYT-X40s — Evaluation & Metrics
===================================
Implements:
  1. Polygon IoU via Shapely (with buffer(0) / convex-hull fallback for topological safety)
  2. Instance-level Precision, Recall, F1 (IoU >= 0.5)
  3. Paper's raster pixel-area overlap metric (Sec 4.1, Eqs 8-10)
  4. Confidence threshold sweep (best-F1 finder)
  5. Detection / E2E H-mean calculation
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
from shapely.geometry import Polygon as ShapelyPoly, MultiPoint
from tqdm import tqdm


def polygon_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """
    Computes IoU between two polygons using Shapely with fast AABB bounding box pre-filtering.
    Topologically safe: handles self-intersecting or degenerated polygons
    using buffer(0) or convex hull fallback.
    """
    if len(poly1) < 3 or len(poly2) < 3:
        return 0.0

    min1 = poly1.min(axis=0)
    max1 = poly1.max(axis=0)
    min2 = poly2.min(axis=0)
    max2 = poly2.max(axis=0)

    if (max1[0] < min2[0] or min1[0] > max2[0] or
        max1[1] < min2[1] or min1[1] > max2[1]):
        return 0.0

    try:
        p1 = ShapelyPoly(poly1)
        p2 = ShapelyPoly(poly2)

        if not p1.is_valid:
            p1 = p1.buffer(0)
            if not p1.is_valid:
                p1 = MultiPoint(poly1).convex_hull

        if not p2.is_valid:
            p2 = p2.buffer(0)
            if not p2.is_valid:
                p2 = MultiPoint(poly2).convex_hull

        if p1.area <= 1e-8 or p2.area <= 1e-8:
            return 0.0

        inter = p1.intersection(p2).area
        union = p1.area + p2.area - inter
        return float(inter / union) if union > 1e-8 else 0.0
    except Exception:
        return 0.0


def evaluate_instance_metrics(
    all_preds: List[np.ndarray],
    all_gts: List[np.ndarray],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Matches predictions to ground truth polygons greedily per image.
    Returns precision, recall, f1, tp, fp, fn.
    """
    tp = fp = fn = 0

    for preds, gts in zip(all_preds, all_gts):
        num_preds = len(preds)
        num_gts   = len(gts)

        if num_gts == 0:
            fp += num_preds
            continue
        if num_preds == 0:
            fn += num_gts
            continue

        iou_matrix = np.zeros((num_preds, num_gts), dtype=np.float32)
        for i, p in enumerate(preds):
            for j, g in enumerate(gts):
                iou_matrix[i, j] = polygon_iou(p, g)

        matched_gt = set()
        matched_pred = set()

        if num_preds > 0 and num_gts > 0:
            flat_indices = np.argsort(-iou_matrix, axis=None)
            for flat_idx in flat_indices:
                p_idx, g_idx = np.unravel_index(flat_idx, iou_matrix.shape)
                if iou_matrix[p_idx, g_idx] < iou_threshold:
                    break
                if p_idx not in matched_pred and g_idx not in matched_gt:
                    matched_pred.add(p_idx)
                    matched_gt.add(g_idx)

        current_tp = len(matched_gt)
        tp += current_tp
        fp += num_preds - current_tp
        fn += num_gts - current_tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)

    return {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "tp":        int(tp),
        "fp":        int(fp),
        "fn":        int(fn),
    }


@torch.no_grad()
def evaluate_area_metric(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    conf_thresh: float = 0.3,
    render_size: int = 640,
) -> Dict[str, float]:
    """
    Paper's pixel-area overlap metric (Eqs 8-10):
    P = sum|pred & GT| / sum|pred|, R = sum|pred & GT| / sum|GT|, F = 2PR / (P+R)
    """
    model.eval()
    inter = pred_area = gt_area = 0.0

    for imgs, targets in loader:
        imgs = imgs.to(device, dtype=torch.float32)
        out = model(imgs)
        scores = torch.sigmoid(out["pred_logits"][:, :, 0]).cpu().numpy()
        coords = out["pred_coords"].cpu().numpy()

        for b in range(len(targets)):
            pm = np.zeros((render_size, render_size), np.uint8)
            gm = np.zeros((render_size, render_size), np.uint8)

            keep = scores[b] > conf_thresh
            for poly in coords[b][keep]:
                pts = (poly * render_size).astype(np.int32)
                cv2.fillPoly(pm, [pts], 1)

            vm = targets[b]["valid_mask"].cpu().numpy()
            for poly in targets[b]["polygons"].cpu().numpy()[vm]:
                pts = (poly * render_size).astype(np.int32)
                cv2.fillPoly(gm, [pts], 1)

            inter += float((pm & gm).sum())
            pred_area += float(pm.sum())
            gt_area += float(gm.sum())

    P = inter / (pred_area + 1e-9)
    R = inter / (gt_area + 1e-9)
    F = 2 * P * R / (P + R + 1e-9) if (P + R) > 0 else 0.0

    return {
        "area_precision": float(P),
        "area_recall":    float(R),
        "area_f1":        float(F),
    }


@torch.no_grad()
def sweep_confidence_thresholds(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    thresholds: Optional[List[float]] = None,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Caches model predictions once with AMP acceleration and sweeps confidence thresholds.
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    cached = []
    use_amp = device.type == "cuda"

    print("Caching predictions for confidence sweep...")
    for images, targets in tqdm(loader, desc="Eval Cache", leave=False):
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            outputs = model(images)
            scores = torch.sigmoid(outputs["pred_logits"][:, :, 0]).cpu().numpy()
            coords = outputs["pred_coords"].cpu().numpy()

        for b in range(len(targets)):
            vm = targets[b]["valid_mask"].cpu().numpy()
            gts = targets[b]["polygons"].cpu().numpy()[vm]
            cached.append((scores[b], coords[b], gts))

    best = {"f1": -1.0, "threshold": 0.3}
    all_results = []

    for thr in thresholds:
        all_preds, all_gts = [], []
        for sc, co, gts in cached:
            keep = sc > thr
            all_preds.append(co[keep])
            all_gts.append(gts)

        m = evaluate_instance_metrics(all_preds, all_gts, iou_threshold=iou_threshold)
        m["threshold"] = thr
        all_results.append(m)

        if m["f1"] > best["f1"]:
            best = m

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "best": best,
        "all": all_results,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: Any,
    device: torch.device,
    conf_thresh: float = 0.3,
    iou_thresh: float = 0.5,
    render_size: int = 256,
) -> Dict[str, float]:
    """
    Runs unified, AMP-accelerated evaluation: instance metrics + paper area metric in a single pass.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.eval()
    all_preds, all_gts = [], []
    inter, pred_area, gt_area = 0.0, 0.0, 0.0

    use_amp = device.type == "cuda"
    for images, targets in loader:
        images = images.to(device, dtype=torch.float32, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=torch.float16):
            outputs = model(images)
            scores = torch.sigmoid(outputs["pred_logits"][:, :, 0]).cpu().numpy()
            coords = outputs["pred_coords"].cpu().numpy()

        for b in range(len(targets)):
            keep = scores[b] > conf_thresh
            preds_b = coords[b][keep]
            vm = targets[b]["valid_mask"].cpu().numpy()
            gts_b = targets[b]["polygons"].cpu().numpy()[vm]

            all_preds.append(preds_b)
            all_gts.append(gts_b)

            pm = np.zeros((render_size, render_size), np.uint8)
            gm = np.zeros((render_size, render_size), np.uint8)

            for poly in preds_b:
                pts = (poly * render_size).astype(np.int32)
                cv2.fillPoly(pm, [pts], 1)

            for poly in gts_b:
                pts = (poly * render_size).astype(np.int32)
                cv2.fillPoly(gm, [pts], 1)

            inter += float((pm & gm).sum())
            pred_area += float(pm.sum())
            gt_area += float(gm.sum())

    inst_metrics = evaluate_instance_metrics(all_preds, all_gts, iou_threshold=iou_thresh)

    P = inter / (pred_area + 1e-9)
    R = inter / (gt_area + 1e-9)
    F = 2 * P * R / (P + R + 1e-9) if (P + R) > 0 else 0.0

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        **inst_metrics,
        "area_precision": float(P),
        "area_recall": float(R),
        "area_f1": float(F),
    }
