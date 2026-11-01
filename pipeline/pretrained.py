"""
DIPCYT-X40s — Pretrained Weight Transfer
=========================================
Transfers pretrained YOLOv5 weights into TPHYOLOv5PlusPlus backbone.
Maps BatchNorm weights/biases directly into GroupNorm parameters.
Internal bottleneck weights in C3 blocks are properly extracted and mapped.
"""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile
import urllib.request
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any, Optional

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .model import TPHYOLOv5PlusPlus



def _slice_conv(yl_weight: torch.Tensor, our_weight: torch.Tensor) -> torch.Tensor:
    Co, Ci = our_weight.shape[:2]
    k_ours = our_weight.shape[2:]
    k_yl   = yl_weight.shape[2:]
    assert k_ours == k_yl, f"Kernel mismatch: ours={k_ours} vs yl={k_yl}"
    assert yl_weight.shape[0] >= Co, f"YOLOv5 out_c={yl_weight.shape[0]} < ours={Co}"
    assert yl_weight.shape[1] >= Ci, f"YOLOv5 in_c={yl_weight.shape[1]} < ours={Ci}"
    return yl_weight[:Co, :Ci].detach().cpu().clone()


def _slice_norm(yl_param: torch.Tensor, our_param: torch.Tensor) -> torch.Tensor:
    Co = our_param.shape[0]
    assert yl_param.shape[0] >= Co, \
        f"YOLOv5 norm param size {yl_param.shape[0]} < ours {Co}"
    return yl_param[:Co].detach().cpu().clone()



def _load_yolov5_state_dict(cache_path: Path) -> dict:
    url = "https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt"

    if not cache_path.exists():
        print(f"  Downloading yolov5s.pt -> {cache_path} ...")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
        print(f"  [OK] Downloaded ({cache_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"  [OK] Cache hit ({cache_path.stat().st_size / 1e6:.1f} MB)")

    repo_dir = Path(tempfile.gettempdir()) / "yolov5"
    if not repo_dir.exists():
        try:
            print("  Cloning ultralytics/yolov5 (depth=1) for model deserialization ...")
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/ultralytics/yolov5.git", str(repo_dir)],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            print("  [OK] Cloned to temp/yolov5")
        except Exception as e:
            print(f"  [WARNING] Git clone skipped or failed: {e}. Trying direct load.")

    repo_str = str(repo_dir)
    if repo_dir.exists() and repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    print("  Loading checkpoint ...")
    ckpt = torch.load(cache_path, map_location="cpu", weights_only=False)
    raw_sd = ckpt["model"].float().state_dict()
    print(f"  [OK] State dict loaded - {len(raw_sd)} keys")
    return raw_sd


def _extract_yolov5_weights(raw_sd: dict) -> dict:
    """Extract all transferable weights from YOLOv5 state dict."""
    def get(key):
        if key not in raw_sd:
            nearby = [k for k in raw_sd if k.startswith(".".join(key.split(".")[:2]))]
            raise KeyError(f"Key not found: {key!r}\n  Nearby keys: {nearby[:10]}")
        return raw_sd[key]

    result = {
        "pconv1_conv":   get("model.1.conv.weight"),
        "c3_1_cv1":      get("model.2.cv1.conv.weight"),
        "c3_1_cv2":      get("model.2.cv2.conv.weight"),
        "c3_1_cv3":      get("model.2.cv3.conv.weight"),
        "pconv2_conv":   get("model.3.conv.weight"),
        "c3_2_cv1":      get("model.4.cv1.conv.weight"),
        "c3_2_cv2":      get("model.4.cv2.conv.weight"),
        "c3_2_cv3":      get("model.4.cv3.conv.weight"),
        "conv1":         get("model.5.conv.weight"),
        "c3_3_cv1":      get("model.6.cv1.conv.weight"),
        "c3_3_cv2":      get("model.6.cv2.conv.weight"),
        "c3_3_cv3":      get("model.6.cv3.conv.weight"),
        "conv2":         get("model.7.conv.weight"),
        "sppf_cv1":      get("model.9.cv1.conv.weight"),
        "sppf_cv2":      get("model.9.cv2.conv.weight"),

        "pconv1_bn_w":   get("model.1.bn.weight"),
        "c3_1_cv1_bn_w": get("model.2.cv1.bn.weight"),
        "c3_1_cv2_bn_w": get("model.2.cv2.bn.weight"),
        "c3_1_cv3_bn_w": get("model.2.cv3.bn.weight"),
        "pconv2_bn_w":   get("model.3.bn.weight"),
        "c3_2_cv1_bn_w": get("model.4.cv1.bn.weight"),
        "c3_2_cv2_bn_w": get("model.4.cv2.bn.weight"),
        "c3_2_cv3_bn_w": get("model.4.cv3.bn.weight"),
        "conv1_bn_w":    get("model.5.bn.weight"),
        "c3_3_cv1_bn_w": get("model.6.cv1.bn.weight"),
        "c3_3_cv2_bn_w": get("model.6.cv2.bn.weight"),
        "c3_3_cv3_bn_w": get("model.6.cv3.bn.weight"),
        "conv2_bn_w":    get("model.7.bn.weight"),
        "sppf_cv1_bn_w": get("model.9.cv1.bn.weight"),
        "sppf_cv2_bn_w": get("model.9.cv2.bn.weight"),

        "pconv1_bn_b":   get("model.1.bn.bias"),
        "c3_1_cv1_bn_b": get("model.2.cv1.bn.bias"),
        "c3_1_cv2_bn_b": get("model.2.cv2.bn.bias"),
        "c3_1_cv3_bn_b": get("model.2.cv3.bn.bias"),
        "pconv2_bn_b":   get("model.3.bn.bias"),
        "c3_2_cv1_bn_b": get("model.4.cv1.bn.bias"),
        "c3_2_cv2_bn_b": get("model.4.cv2.bn.bias"),
        "c3_2_cv3_bn_b": get("model.4.cv3.bn.bias"),
        "conv1_bn_b":    get("model.5.bn.bias"),
        "c3_3_cv1_bn_b": get("model.6.cv1.bn.bias"),
        "c3_3_cv2_bn_b": get("model.6.cv2.bn.bias"),
        "c3_3_cv3_bn_b": get("model.6.cv3.bn.bias"),
        "conv2_bn_b":    get("model.7.bn.bias"),
        "sppf_cv1_bn_b": get("model.9.cv1.bn.bias"),
        "sppf_cv2_bn_b": get("model.9.cv2.bn.bias"),
    }

    c3_configs = [
        ("c3_1", 2, 3),    # model.2, 3 bottleneck blocks
        ("c3_2", 4, 9),    # model.4, 9 bottleneck blocks
        ("c3_3", 6, 9),    # model.6, 9 bottleneck blocks
    ]

    for name, layer_idx, num_blocks in c3_configs:
        for bi in range(num_blocks):
            for ci in [1, 2]:
                prefix = f"model.{layer_idx}.m.{bi}.cv{ci}"
                try:
                    result[f"{name}_m{bi}_cv{ci}"]      = get(f"{prefix}.conv.weight")
                    result[f"{name}_m{bi}_cv{ci}_bn_w"]  = get(f"{prefix}.bn.weight")
                    result[f"{name}_m{bi}_cv{ci}_bn_b"]  = get(f"{prefix}.bn.bias")
                except KeyError:
                    pass

    return result



def load_yolov5l_backbone_weights(
    backbone: "TPHYOLOv5PlusPlus",
    cache_path: str | Path = "yolov5s.pt",
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Transfers YOLOv5 pretrained weights to TPHYOLOv5PlusPlus.
    Returns dictionary of transfer statistics.
    """
    cache_path = Path(cache_path)
    raw_sd = _load_yolov5_state_dict(cache_path)
    yl_w   = _extract_yolov5_weights(raw_sd)

    bb_total_params = sum(p.numel() for p in backbone.parameters())

    CONV_MAPPING = [
        ("pconv1.pconv.conv", yl_w["pconv1_conv"]),
        ("c3_1.cv1.conv",     yl_w["c3_1_cv1"]),
        ("c3_1.cv2.conv",     yl_w["c3_1_cv2"]),
        ("c3_1.cv3.conv",     yl_w["c3_1_cv3"]),
        ("pconv2.pconv.conv", yl_w["pconv2_conv"]),
        ("c3_2.cv1.conv",     yl_w["c3_2_cv1"]),
        ("c3_2.cv2.conv",     yl_w["c3_2_cv2"]),
        ("c3_2.cv3.conv",     yl_w["c3_2_cv3"]),
        ("conv1.conv",        yl_w["conv1"]),
        ("c3_3.cv1.conv",     yl_w["c3_3_cv1"]),
        ("c3_3.cv2.conv",     yl_w["c3_3_cv2"]),
        ("c3_3.cv3.conv",     yl_w["c3_3_cv3"]),
        ("conv2.conv",        yl_w["conv2"]),
        ("sppf.cv1.conv",     yl_w["sppf_cv1"]),
        ("sppf.cv2.conv",     yl_w["sppf_cv2"]),
    ]

    NORM_MAPPING = [
        ("pconv1.pconv.norm", yl_w["pconv1_bn_w"],   yl_w["pconv1_bn_b"]),
        ("c3_1.cv1.norm",     yl_w["c3_1_cv1_bn_w"], yl_w["c3_1_cv1_bn_b"]),
        ("c3_1.cv2.norm",     yl_w["c3_1_cv2_bn_w"], yl_w["c3_1_cv2_bn_b"]),
        ("c3_1.cv3.norm",     yl_w["c3_1_cv3_bn_w"], yl_w["c3_1_cv3_bn_b"]),
        ("pconv2.pconv.norm", yl_w["pconv2_bn_w"],   yl_w["pconv2_bn_b"]),
        ("c3_2.cv1.norm",     yl_w["c3_2_cv1_bn_w"], yl_w["c3_2_cv1_bn_b"]),
        ("c3_2.cv2.norm",     yl_w["c3_2_cv2_bn_w"], yl_w["c3_2_cv2_bn_b"]),
        ("c3_2.cv3.norm",     yl_w["c3_2_cv3_bn_w"], yl_w["c3_2_cv3_bn_b"]),
        ("conv1.norm",        yl_w["conv1_bn_w"],    yl_w["conv1_bn_b"]),
        ("c3_3.cv1.norm",     yl_w["c3_3_cv1_bn_w"], yl_w["c3_3_cv1_bn_b"]),
        ("c3_3.cv2.norm",     yl_w["c3_3_cv2_bn_w"], yl_w["c3_3_cv2_bn_b"]),
        ("c3_3.cv3.norm",     yl_w["c3_3_cv3_bn_w"], yl_w["c3_3_cv3_bn_b"]),
        ("conv2.norm",        yl_w["conv2_bn_w"],    yl_w["conv2_bn_b"]),
        ("sppf.cv1.norm",     yl_w["sppf_cv1_bn_w"], yl_w["sppf_cv1_bn_b"]),
        ("sppf.cv2.norm",     yl_w["sppf_cv2_bn_w"], yl_w["sppf_cv2_bn_b"]),
    ]

    c3_configs = [
        ("c3_1", 3),    # 3 bottleneck blocks
        ("c3_2", 9),    # 9 bottleneck blocks
        ("c3_3", 9),    # 9 bottleneck blocks
    ]

    for name, num_blocks in c3_configs:
        for bi in range(num_blocks):
            for ci in [1, 2]:
                conv_key = f"{name}_m{bi}_cv{ci}"
                bn_w_key = f"{name}_m{bi}_cv{ci}_bn_w"
                bn_b_key = f"{name}_m{bi}_cv{ci}_bn_b"

                attr_path = f"{name}.blocks.{bi}.cv{ci}.conv"
                if conv_key in yl_w:
                    CONV_MAPPING.append((attr_path, yl_w[conv_key]))

                norm_path = f"{name}.blocks.{bi}.cv{ci}.norm"
                if bn_w_key in yl_w and bn_b_key in yl_w:
                    NORM_MAPPING.append((norm_path, yl_w[bn_w_key], yl_w[bn_b_key]))

    our_sd = backbone.state_dict()
    new_sd = {k: v.clone() for k, v in our_sd.items()}
    stats  = dict(conv_sliced=0, norm_sliced=0, skipped=0, errors=0, sliced_mismatch=0)

    for attr_path, yl_weight in CONV_MAPPING:
        key = f"{attr_path}.weight"
        if key not in new_sd:
            stats["skipped"] += 1
            continue
        try:
            our_w = new_sd[key]
            sliced = _slice_conv(yl_weight, our_w)
            if tuple(our_w.shape) != tuple(yl_weight.shape):
                stats["sliced_mismatch"] += 1
            new_sd[key] = sliced
            stats["conv_sliced"] += 1
        except Exception as exc:
            if verbose:
                print(f"  [ERROR] Error transferring {key}: {exc}")
            stats["errors"] += 1

    for attr_path, bn_gamma, bn_beta in NORM_MAPPING:
        w_key = f"{attr_path}.weight"
        b_key = f"{attr_path}.bias"
        if w_key not in new_sd or b_key not in new_sd:
            stats["skipped"] += 1
            continue
        try:
            sliced_w = _slice_norm(bn_gamma, new_sd[w_key])
            sliced_b = _slice_norm(bn_beta,  new_sd[b_key])
            if new_sd[w_key].shape[0] != bn_gamma.shape[0]:
                stats["sliced_mismatch"] += 1
            new_sd[w_key] = sliced_w
            new_sd[b_key] = sliced_b
            stats["norm_sliced"] += 1
        except Exception as exc:
            if verbose:
                print(f"  [ERROR] Error transferring {attr_path}: {exc}")
            stats["errors"] += 1

    backbone.load_state_dict(new_sd)

    any_nan = any(p.isnan().any().item() for p in backbone.parameters())
    any_inf = any(p.isinf().any().item() for p in backbone.parameters())
    if any_nan or any_inf:
        raise ValueError("Transferred backbone contains NaN or Inf!")

    transferred_conv = sum(
        new_sd[f"{ap}.weight"].numel()
        for ap, _ in CONV_MAPPING
        if f"{ap}.weight" in new_sd
    )
    transferred_norm = sum(
        new_sd[f"{ap}.weight"].numel() + new_sd[f"{ap}.bias"].numel()
        for ap, _, _ in NORM_MAPPING
        if f"{ap}.weight" in new_sd
    )
    total_transferred = transferred_conv + transferred_norm
    coverage = total_transferred / bb_total_params
    stats["coverage"] = coverage
    stats["total_transferred"] = total_transferred
    stats["bb_total_params"] = bb_total_params

    print(f"  [OK] Pretrained transfer: {total_transferred:,} / {bb_total_params:,} ({100*coverage:.1f}% coverage)")
    return stats
