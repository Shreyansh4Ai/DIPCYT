"""
DIPCYT-X40s — Loss Functions & Hungarian Matcher
=================================================
Implements 4-head size-bucketed Hungarian assignment and composite loss:
  L = λ_coord * L_coord + λ_giou * L_giou + λ_conf * L_focal + λ_shape * L_shape

Guards:
  - Size-bucketed matching matches scale-specialised heads (D1-D4) to appropriate GT sizes.
  - Class-balanced focal loss normalises positive and negative queries independently.
  - Convexity regularisation penalty prevents self-intersecting polygon degeneracies.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from .config import LossConfig, MatcherConfig


def _poly_to_box(polys: torch.Tensor) -> torch.Tensor:
    """Extracts axis-aligned bounding box [xmin, ymin, xmax, ymax] from polygon vertices."""
    xmin = polys[..., 0].min(dim=-1)[0]
    ymin = polys[..., 1].min(dim=-1)[0]
    xmax = polys[..., 0].max(dim=-1)[0]
    ymax = polys[..., 1].max(dim=-1)[0]
    return torch.stack([xmin, ymin, xmax, ymax], dim=-1)


def _box_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """Converts [x1, y1, x2, y2] to [cx, cy, w, h]."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1), (y2 - y1)], dim=-1)


def _pairwise_giou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise Generalized IoU between N boxes and M boxes."""
    b1_x1, b1_y1, b1_x2, b1_y2 = boxes1.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = boxes2.unbind(-1)

    inter_x1 = torch.max(b1_x1[:, None], b2_x1[None, :])
    inter_y1 = torch.max(b1_y1[:, None], b2_y1[None, :])
    inter_x2 = torch.min(b1_x2[:, None], b2_x2[None, :])
    inter_y2 = torch.min(b1_y2[:, None], b2_y2[None, :])

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)

    enc_x1 = torch.min(b1_x1[:, None], b2_x1[None, :])
    enc_y1 = torch.min(b1_y1[:, None], b2_y1[None, :])
    enc_x2 = torch.max(b1_x2[:, None], b2_x2[None, :])
    enc_y2 = torch.max(b1_y2[:, None], b2_y2[None, :])
    enc_area = torch.clamp(enc_x2 - enc_x1, min=0) * torch.clamp(enc_y2 - enc_y1, min=0)

    giou = iou - (enc_area - union_area) / torch.clamp(enc_area, min=1e-6)
    return giou


def _giou_loss_matched(pred_coords: torch.Tensor, gt_coords: torch.Tensor) -> torch.Tensor:
    """GIoU loss (1 - GIoU) for matched prediction and ground truth polygons."""
    pred_boxes = _poly_to_box(pred_coords)
    gt_boxes   = _poly_to_box(gt_coords)
    b1_x1, b1_y1, b1_x2, b1_y2 = pred_boxes.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = gt_boxes.unbind(-1)

    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    union_area = area1 + area2 - inter_area
    iou = inter_area / torch.clamp(union_area, min=1e-6)

    enc_x1 = torch.min(b1_x1, b2_x1)
    enc_y1 = torch.min(b1_y1, b2_y1)
    enc_x2 = torch.max(b1_x2, b2_x2)
    enc_y2 = torch.max(b1_y2, b2_y2)
    enc_area = torch.clamp(enc_x2 - enc_x1, min=0) * torch.clamp(enc_y2 - enc_y1, min=0)

    giou = iou - (enc_area - union_area) / torch.clamp(enc_area, min=1e-6)
    return (1.0 - giou).mean()


def _polygon_shape_penalty(polys: torch.Tensor, point_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Penalises non-convex / self-intersecting polygon turns."""
    M, P, _ = polys.shape
    if M == 0 or P < 3:
        return polys.sum() * 0.0
    x, y = polys[..., 0], polys[..., 1]
    x1, y1 = torch.roll(x, -1, dims=1), torch.roll(y, -1, dims=1)
    xm1, ym1 = torch.roll(x, 1, dims=1), torch.roll(y, 1, dims=1)
    e1x, e1y = x - xm1, y - ym1
    e2x, e2y = x1 - x, y1 - y
    cross = e1x * e2y - e1y * e2x
    majority = torch.sign(cross.sum(dim=1, keepdim=True))
    disagree = (torch.sign(cross) != majority).float()
    if point_mask is not None:
        pm = point_mask.float()
        disagree = disagree * pm
        denom = pm.sum(dim=1).clamp(min=1.0)
    else:
        denom = torch.tensor(float(P), device=polys.device)
    return (disagree.sum(dim=1) / denom).mean()


class PolygonHungarianLoss(nn.Module):
    """
    Hungarian matching loss for 4 multi-scale polygon detection heads (D1-D4).
    Uses scale buckets to route GT instances to the architecturally appropriate head.
    """
    def __init__(
        self,
        loss_cfg: Optional[LossConfig] = None,
        matcher_cfg: Optional[MatcherConfig] = None,
    ):
        super().__init__()
        loss_cfg = loss_cfg or LossConfig()
        matcher_cfg = matcher_cfg or MatcherConfig()

        self.lambda_coord = loss_cfg.lambda_coord
        self.lambda_giou  = loss_cfg.lambda_giou
        self.lambda_conf  = loss_cfg.lambda_conf
        self.lambda_shape = loss_cfg.lambda_shape
        self.focal_alpha  = loss_cfg.focal_alpha
        self.focal_gamma  = loss_cfg.focal_gamma

        self.center_cost_weight = matcher_cfg.center_cost_weight
        self.size_cost_weight   = matcher_cfg.size_cost_weight
        self.giou_cost_weight   = matcher_cfg.giou_cost_weight
        self.bucket_thresholds  = matcher_cfg.bucket_thresholds

        self.reset_log()

    def reset_log(self) -> None:
        self._log_coord = 0.0
        self._log_conf  = 0.0
        self._log_pos   = 0.0
        self._log_neg   = 0.0
        self._log_n     = 0

    def get_log(self) -> Tuple[float, float, float, float]:
        n = max(self._log_n, 1)
        return (self._log_coord / n, self._log_conf / n,
                self._log_pos / n, self._log_neg / n)

    @staticmethod
    def sigmoid_focal_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.75,
        gamma: float = 1.0,
    ) -> torch.Tensor:
        """Class-balanced focal loss: normalises positive and negative queries independently."""
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        p_t = p * targets + (1.0 - p) * (1.0 - targets)
        focal = ce_loss * ((1.0 - p_t) ** gamma)

        pos_mask = (targets > 0.5)
        neg_mask = ~pos_mask
        n_pos = pos_mask.sum().clamp(min=1.0)
        n_neg = neg_mask.sum().clamp(min=1.0)

        pos_loss = (focal * pos_mask).sum() / n_pos
        neg_loss = (focal * neg_mask).sum() / n_neg
        return alpha * pos_loss + (1.0 - alpha) * neg_loss

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: List[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        pred_logits = outputs["pred_logits"]   # (B, 4Q, 1)
        pred_coords = outputs["pred_coords"]   # (B, 4Q, NP, 2)
        ref_centers = outputs["ref_centers"]   # (B, 4Q, 2)
        if ref_centers.dim() == 4:
            ref_centers = ref_centers.mean(dim=2)

        B = pred_logits.shape[0]
        Q = pred_logits.shape[1]              # 4 * queries_per_head
        Q_PER_HEAD = Q // 4

        total_loss  = pred_coords.sum() * 0.0
        batch_coord = 0.0
        batch_conf  = 0.0
        batch_pos_c = 0.0
        batch_neg_c = 0.0

        for i in range(B):
            gt_polys   = targets[i]["polygons"].to(pred_logits.device, dtype=torch.float32)
            valid_mask = targets[i]["valid_mask"].to(pred_logits.device)
            point_mask = targets[i]["point_mask"].to(pred_logits.device)

            with torch.no_grad():
                pred_prob = pred_logits[i, :, 0].sigmoid().clamp(min=1e-6, max=1 - 1e-6)
                pred_boxes = _poly_to_box(pred_coords[i])
                gt_boxes   = _poly_to_box(gt_polys)
                valid_gt_idx = valid_mask.nonzero(as_tuple=True)[0]

                row_ind_t = torch.empty(0, dtype=torch.long, device=pred_logits.device)
                col_ind_t = torch.empty(0, dtype=torch.long, device=pred_logits.device)

                if len(valid_gt_idx) > 0:
                    pred_cxcywh = _box_cxcywh(pred_boxes)
                    gt_cxcywh   = _box_cxcywh(gt_boxes[valid_gt_idx])

                    pred_c = ref_centers[i]                          # (Q, 2)
                    gt_c   = gt_cxcywh[:, :2]                        # (Gv, 2)
                    pred_s = pred_cxcywh[:, 2:]                      # (Q, 2)
                    gt_s   = gt_cxcywh[:, 2:]                        # (Gv, 2)


                    gt_boxes_valid = gt_boxes[valid_gt_idx]
                    gt_w = (gt_boxes_valid[:, 2] - gt_boxes_valid[:, 0]).clamp(min=0)
                    gt_h = (gt_boxes_valid[:, 3] - gt_boxes_valid[:, 1]).clamp(min=0)
                    max_size = torch.max(gt_w, gt_h)

                    Gv = len(valid_gt_idx)
                    head_of_gt = torch.zeros(Gv, dtype=torch.long, device=pred_logits.device)

                    t0, t1, t2 = self.bucket_thresholds
                    head_of_gt[(max_size >= t0) & (max_size < t1)] = 1   # D2
                    head_of_gt[(max_size >= t1) & (max_size < t2)] = 2   # D3
                    head_of_gt[(max_size >= t2)]                   = 3   # D4


                    all_rows, all_cols = [], []
                    for head_idx in range(4):
                        head_gt_local = (head_of_gt == head_idx).nonzero(as_tuple=True)[0]
                        if len(head_gt_local) == 0:
                            continue
                        q_start = head_idx * Q_PER_HEAD
                        q_end   = q_start + Q_PER_HEAD

                        center_cost = (pred_c[q_start:q_end].unsqueeze(1)
                                       - gt_c[head_gt_local].unsqueeze(0)).abs().sum(-1)
                        size_cost   = (pred_s[q_start:q_end].unsqueeze(1)
                                       - gt_s[head_gt_local].unsqueeze(0)).abs().sum(-1)
                        giou_cost   = 1.0 - _pairwise_giou(
                            pred_boxes[q_start:q_end], gt_boxes_valid[head_gt_local])

                        cost_matrix = (self.center_cost_weight * center_cost
                                       + self.size_cost_weight * size_cost
                                       + self.giou_cost_weight * giou_cost)

                        cost_np = cost_matrix.cpu().float().numpy()
                        r, c = linear_sum_assignment(cost_np)
                        all_rows.append(torch.as_tensor(r, device=pred_logits.device) + q_start)
                        all_cols.append(valid_gt_idx[head_gt_local[
                            torch.as_tensor(c, device=pred_logits.device)]])

                    if all_rows:
                        row_ind_t = torch.cat(all_rows)
                        col_ind_t = torch.cat(all_cols)


            valid_col_mask = valid_mask[col_ind_t]
            matched_preds  = pred_coords[i][row_ind_t]
            matched_gts    = gt_polys[col_ind_t]
            matched_pmask  = point_mask[col_ind_t]

            coord_err   = (matched_preds - matched_gts).abs()
            valid_poly  = valid_col_mask.float().view(-1, 1, 1)
            valid_pts   = matched_pmask.float().unsqueeze(-1)
            combined    = valid_poly * valid_pts
            num_real_pts = combined.sum().clamp(min=1.0)

            loss_coord = (coord_err * combined).sum() / num_real_pts

            if valid_col_mask.any():
                loss_giou  = _giou_loss_matched(
                    matched_preds[valid_col_mask], matched_gts[valid_col_mask])
                loss_shape = _polygon_shape_penalty(
                    matched_preds[valid_col_mask], matched_pmask[valid_col_mask])
            else:
                loss_giou  = loss_coord * 0.0
                loss_shape = loss_coord * 0.0

            conf_target = torch.zeros(Q, device=pred_logits.device)
            valid_rows  = row_ind_t[valid_col_mask]
            conf_target[valid_rows] = 1.0

            loss_conf = self.sigmoid_focal_loss(
                pred_logits[i, :, 0], conf_target,
                alpha=self.focal_alpha, gamma=self.focal_gamma)

            img_loss = (self.lambda_coord * loss_coord
                        + self.lambda_giou * loss_giou
                        + self.lambda_conf * loss_conf
                        + self.lambda_shape * loss_shape)
            total_loss = total_loss + img_loss * valid_mask.any().float()

            batch_coord += loss_coord.item()
            batch_conf  += loss_conf.item()

            with torch.no_grad():
                if valid_rows.numel() > 0:
                    batch_pos_c += pred_prob[valid_rows].mean().item()
                    neg_mask = torch.ones(Q, dtype=torch.bool, device=pred_logits.device)
                    neg_mask[valid_rows] = False
                    batch_neg_c += (pred_prob[neg_mask].mean().item()
                                    if neg_mask.any() else 0.0)
                else:
                    batch_neg_c += pred_prob.mean().item()

        self._log_coord += batch_coord / B
        self._log_conf  += batch_conf / B
        self._log_pos   += batch_pos_c / B
        self._log_neg   += batch_neg_c / B
        self._log_n     += 1

        return total_loss / B
