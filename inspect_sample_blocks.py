"""
Pass a real COCO-Text image sample through every block of DIPCYT-X40s
and inspect intermediate tensors, shapes, statistics, and outputs.
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from pipeline.config import Config
from pipeline.dataset import build_loaders
from pipeline.model import FullModel


def format_stats(t: torch.Tensor, name: str = "") -> str:
    if not isinstance(t, torch.Tensor):
        return str(type(t))
    tf = t.detach().float()
    return (
        f"{name:<22} | Shape: {str(list(t.shape)):<22} | "
        f"Min: {tf.min().item():+8.4f} | Max: {tf.max().item():+8.4f} | "
        f"Mean: {tf.mean().item():+8.4f} | Std: {tf.std().item():8.4f}"
    )


def print_header(title: str):
    print("\n" + "=" * 95)
    print(f"  {title}")
    print("=" * 95)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    cfg = Config.from_yaml("extended_config.yaml")
    cfg.model.pretrained_backbone = False  # loading full checkpoint
    cfg.hardware.pin_memory = False
    cfg.hardware.num_workers = 0

    print("[*] Instantiating DIPCYT-X40s FullModel...")
    model = FullModel(cfg.model)

    ckpt_path = "checkpoints/best_model.pth"
    print(f"[*] Loading trained weights from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()

    print("[*] Loading COCO-Text dataset to obtain a sample image...")
    train_loader, val_loader, test_loader, ann_store = build_loaders(cfg)

    sample_img = None
    sample_tgt = None
    sample_idx = 0
    for i in range(len(train_loader.dataset)):
        s_img, s_tgt = train_loader.dataset[i]
        if s_tgt["valid_mask"].sum().item() > 0:
            sample_img = s_img
            sample_tgt = s_tgt
            sample_idx = i
            break

    print(f"[*] Selected Dataset Sample #{sample_idx}:")
    gt_valids = sample_tgt["valid_mask"].sum().item()
    print(f"    - Image Tensor Shape : {sample_img.shape}")
    print(f"    - Original Size (H,W): {sample_tgt['orig_size'].tolist()}")
    print(f"    - Ground-truth Boxes : {gt_valids}")

    x = sample_img.unsqueeze(0).to(device)  # [1, 3, 640, 640]

    out_dir = "block_outputs_inspection"
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        print_header("BLOCK 0: INPUT TENSOR")
        print(format_stats(x, "Input (x)"))
        print("  Handoff: Raw normalized image [1, 3, 640, 640] passed to Backbone.")

        print_header("BLOCK 1: BACKBONE (TPHYOLOv5PlusPlus) — STAGES")
        bb = model.backbone

        x_focus = bb.focus(x)
        print(format_stats(x_focus, "1.1 Focus(x)"))
        print("      Operation: Space-to-depth 4x sub-sampling + 1x1 conv [1, 32, 320, 320]")

        x_pconv1 = bb.pconv1(x_focus)
        x_c3_1 = bb.c3_1(x_pconv1)
        print(format_stats(x_pconv1, "1.2 PConv1"))
        print(format_stats(x_c3_1, "    C3_1 (3 blocks)"))

        x_pconv2 = bb.pconv2(x_c3_1)
        p2 = bb.c3_2(x_pconv2)
        print(format_stats(x_pconv2, "1.3 PConv2"))
        print(format_stats(p2, "    C3_2 -> P2 (stride 8)"))

        x_conv1 = bb.conv1(p2)
        p3 = bb.c3_3(x_conv1)
        print(format_stats(x_conv1, "1.4 Conv1"))
        print(format_stats(p3, "    C3_3 -> P3 (stride 16) [Backbone Out 1]"))

        p4 = bb.conv2(p3)
        print(format_stats(p4, "1.5 Conv2 -> P4 (stride 32) [Backbone Out 2]"))

        x_sppf = bb.sppf(p4)
        p5 = bb.c3tr(x_sppf)
        print(format_stats(x_sppf, "1.6 SPPF"))
        print(format_stats(p5, "    C3TR -> P5 (stride 32) [Backbone Out 3]"))
        print("      C3TR applies 3 Transformer Encoder layers (dim=256, 8 heads) to tokens.")

        print_header("BLOCK 2: PANet — LATERALS & FPCBAM ATTENTION MODULES")
        fpn = model.fpn
        features = [p3, p4, p5]

        lats_raw = [lat(f) for lat, f in zip(fpn.laterals, features)]
        print(format_stats(lats_raw[0], "Lateral P3 -> C"))
        print(format_stats(lats_raw[1], "Lateral P4 -> C"))
        print(format_stats(lats_raw[2], "Lateral P5 -> C"))

        lats = [fpn.td_fpcbam[i](lats_raw[i]) for i in range(3)]
        print(format_stats(lats[0], "FPCBAM(Lat P3)"))
        print(format_stats(lats[1], "FPCBAM(Lat P4) [PAM enabled]"))
        print(format_stats(lats[2], "FPCBAM(Lat P5) [PAM enabled]"))
        print("      FPCBAM: FFT high-pass residual + Channel MLP + Spatial + Positional (PAM).")

        print_header("BLOCK 3: PANet — TOP-DOWN & BOTTOM-UP MULTI-SCALE PYRAMID")
        p5_td = fpn.td_p5_tran(lats[2])
        p5_up = F.interpolate(p5_td, size=lats[1].shape[-2:], mode='nearest')
        p4_fused = fpn.td_fuse[1](torch.cat([lats[1], p5_up], dim=1))
        p4_td = fpn.td_c3[1](p4_fused)
        p4_up = F.interpolate(p4_td, size=lats[0].shape[-2:], mode='nearest')
        p3_fused = fpn.td_fuse[0](torch.cat([lats[0], p4_up], dim=1))
        d1 = fpn.td_c3[0](p3_fused)

        print(format_stats(p5_td, "Top-Down Swin(P5)"))
        print(format_stats(p4_td, "Top-Down C3(P4_fused)"))
        print(format_stats(d1, "Output D1 (stride 16)"))

        p5_td_10 = F.avg_pool2d(p5_td, kernel_size=2, stride=2)
        p5_td_5  = F.avg_pool2d(p5_td_10, kernel_size=2, stride=2)

        d1_down = fpn.bu_convs[0](d1)
        bu1     = fpn.bu_fuse[0](torch.cat([p4_td, d1_down], dim=1))
        bu1     = fpn.bu_tran[0](bu1)
        d2      = fpn.bu_fpcbam[0](bu1)

        d2_down = fpn.bu_convs[1](d2)
        bu2     = fpn.bu_fuse[1](torch.cat([p5_td_10, d2_down], dim=1))
        bu2     = fpn.bu_tran[1](bu2)
        d3      = fpn.bu_fpcbam[1](bu2)

        d3_down = fpn.bu_convs[2](d3)
        bu3     = fpn.bu_fuse[2](torch.cat([p5_td_5, d3_down], dim=1))
        d4      = fpn.bu_tran[2](bu3)

        print(format_stats(d2, "Output D2 (stride 32)"))
        print(format_stats(d3, "Output D3 (stride 64)"))
        print(format_stats(d4, "Output D4 (stride 128)"))

        fpn_outputs = [d1, d2, d3, d4]

        print_header("BLOCK 4: TRANSFORMER POLYGON HEADS (4 LEVELS: D1, D2, D3, D4)")
        logits_list, coords_list, ref_list = [], [], []

        for i, (head, feat) in enumerate(zip(model.heads, fpn_outputs)):
            lvl_name = f"D{i+1}"
            B, C, H, W = feat.shape
            feat_proj = head.proj(feat)
            pos = head.pos_enc(feat)
            feat_flat = feat_proj.flatten(2).transpose(1, 2)
            pos_flat = pos.flatten(2).transpose(1, 2)
            memory = head.encoder(feat_flat + pos_flat)

            tgt = torch.zeros(B, head.num_queries, head.in_channels, device=feat.device)
            q_pos = head.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
            for layer in head.decoder.layers:
                tgt = layer(tgt + q_pos, memory + pos_flat)
            out_dec = tgt

            pred_logits_i = head.conf_head(out_dec)
            ref_i = head.ref_points.unsqueeze(0).sigmoid()
            delta_i = head.poly_head(out_dec) * 0.1
            pred_coords_i = (ref_i + delta_i).clamp(0.0, 1.0).view(B, head.num_queries, head.num_points, 2)
            ref_centers_i = head.ref_points.sigmoid().view(head.num_queries, head.num_points, 2).mean(dim=1).unsqueeze(0).expand(B, -1, -1)

            logits_list.append(pred_logits_i)
            coords_list.append(pred_coords_i)
            ref_list.append(ref_centers_i)

            print(f"\n  --- Head {i} ({lvl_name} feature map: {H}x{W}) ---")
            print(f"      Encoder Tokens Memory : {list(memory.shape)} (Tokens = {H*W})")
            print(format_stats(pred_logits_i, f"      pred_logits_{lvl_name}"))
            print(format_stats(pred_coords_i, f"      pred_coords_{lvl_name}"))
            print(format_stats(ref_centers_i, f"      ref_centers_{lvl_name}"))

            conf = torch.sigmoid(pred_logits_i)[0, :, 0]
            top_conf, top_idx = torch.topk(conf, k=min(3, head.num_queries))
            print(f"      Top-3 Confidences     : {top_conf.cpu().numpy().round(4).tolist()}")

        pred_logits = torch.cat(logits_list, dim=1)  # [1, 128, 1]
        pred_coords = torch.cat(coords_list, dim=1)  # [1, 128, 6, 2]
        ref_centers = torch.cat(ref_list, dim=1)     # [1, 128, 2]

        print_header("FINAL FULL MODEL OUTPUT TENSORS")
        print(format_stats(pred_logits, "pred_logits [Total]"))
        print(format_stats(pred_coords, "pred_coords [Total]"))
        print(format_stats(ref_centers, "ref_centers [Total]"))
        print("  Total queries: 128 (32 queries x 4 scale heads)")

        print_header("GENERATING VISUAL DIAGNOSTIC FIGURES")

        img_np = sample_img.permute(1, 2, 0).cpu().numpy()
        img_np = np.clip(img_np, 0.0, 1.0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(img_np)
        axes[0].set_title(f"Ground Truth Polygons ({gt_valids} boxes)", fontsize=13, fontweight="bold")
        gt_polys = sample_tgt["polygons"]  # [128, 6, 2]
        v_mask = sample_tgt["valid_mask"]
        for p_idx in range(len(v_mask)):
            if v_mask[p_idx]:
                poly = gt_polys[p_idx].cpu().numpy() * 640.0
                patch = patches.Polygon(poly, linewidth=2, edgecolor="lime", facecolor="none")
                axes[0].add_patch(patch)
        axes[0].axis("off")

        axes[1].imshow(img_np)
        all_confs = torch.sigmoid(pred_logits)[0, :, 0].cpu()
        all_coords = (pred_coords[0] * 640.0).cpu().numpy()
        top_k = 8
        top_vals, top_inds = torch.topk(all_confs, k=top_k)
        axes[1].set_title(f"Top-{top_k} Predicted Polygon Candidates", fontsize=13, fontweight="bold")

        colors = ["cyan", "yellow", "magenta", "orange", "deepskyblue", "lime", "red", "gold"]
        for rank, (score, idx) in enumerate(zip(top_vals, top_inds)):
            poly = all_coords[idx]
            patch = patches.Polygon(poly, linewidth=2.2, edgecolor=colors[rank % len(colors)], facecolor="none")
            axes[1].add_patch(patch)
            axes[1].text(
                poly[0, 0], poly[0, 1] - 4,
                f"#{rank+1}: {score.item():.3f}",
                color="white", fontsize=9, fontweight="bold",
                bbox=dict(facecolor=colors[rank % len(colors)], alpha=0.7, pad=1, edgecolor="none")
            )
        axes[1].axis("off")
        plt.tight_layout()
        pred_fig_path = os.path.join(out_dir, "sample_detection_polygons.png")
        plt.savefig(pred_fig_path, dpi=200)
        plt.close()
        print(f"  [Saved] {pred_fig_path}")

        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title("Input Image [640x640]", fontweight="bold")
        axes[0, 0].axis("off")

        def show_feat(ax, feat, title):
            mag = feat[0].mean(dim=0).cpu().numpy()
            im = ax.imshow(mag, cmap="viridis")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        show_feat(axes[0, 1], p3, f"Backbone P3 ({p3.shape[2]}x{p3.shape[3]})")
        show_feat(axes[0, 2], p4, f"Backbone P4 ({p4.shape[2]}x{p4.shape[3]})")
        show_feat(axes[0, 3], p5, f"Backbone P5-C3TR ({p5.shape[2]}x{p5.shape[3]})")

        show_feat(axes[1, 0], d1, f"PANet D1 ({d1.shape[2]}x{d1.shape[3]})")
        show_feat(axes[1, 1], d2, f"PANet D2 ({d2.shape[2]}x{d2.shape[3]})")
        show_feat(axes[1, 2], d3, f"PANet D3 ({d3.shape[2]}x{d3.shape[3]})")
        show_feat(axes[1, 3], d4, f"PANet D4 ({d4.shape[2]}x{d4.shape[3]})")

        plt.suptitle("DIPCYT-X40s: Backbone (P3-P5) vs. PANet Multi-Scale (D1-D4) Activation Maps", fontsize=14, fontweight="bold")
        plt.tight_layout()
        feat_fig_path = os.path.join(out_dir, "sample_feature_pyramids.png")
        plt.savefig(feat_fig_path, dpi=200)
        plt.close()
        print(f"  [Saved] {feat_fig_path}")

        artifact_dir = r"C:\Users\goswa\.gemini\antigravity-ide\brain\8d37f0b6-b19a-4319-bf87-2d241e42e130"
        os.makedirs(artifact_dir, exist_ok=True)
        shutil_dst1 = os.path.join(artifact_dir, "sample_detection_polygons.png")
        shutil_dst2 = os.path.join(artifact_dir, "sample_feature_pyramids.png")
        import shutil
        shutil.copyfile(pred_fig_path, shutil_dst1)
        shutil.copyfile(feat_fig_path, shutil_dst2)
        print(f"  [Copied to Artifacts] {shutil_dst1}")
        print(f"  [Copied to Artifacts] {shutil_dst2}")

        print("\n" + "=" * 95)
        print("  ALL BLOCKS EXECUTED & INSPECTED SUCCESSFULLY!")
        print("=" * 95)


if __name__ == "__main__":
    main()
