"""Instance extraction, boundary erosion, and OOF false-positive filtering."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from .metrics import Mask, mask_from_array
from .rle import mask_to_rle

__all__ = [
    "extract_instances",
    "erode_masks",
    "instances_to_rows",
    "candidate_features",
]

FEATURE_NAMES = [
    "area", "sqrt_area", "bbox_w", "bbox_h", "aspect", "fill",
    "p_mask_mean", "p_mask_max", "p_mask_p90", "p_mask_std",
    "p_hull_mean", "p_hull_max", "p_spine_mean", "p_spine_max",
    "ring_p_mask", "ring_p_hull",
    "elongation", "solidity", "rho",
]


def extract_instances(
    prob_3ch: np.ndarray,
    t_lo: float = 0.35,
    t_hi: float = 0.80,
    min_meanp: float = 0.54,
    hull_thr: float = 0.44,
    spine_thr: float = 0.30,
    min_area: int = 260,
) -> list[np.ndarray]:
    """Extract non-overlapping binary filament masks from (mask, hull, spine) probability maps."""
    p_mask, p_hull, p_spine = prob_3ch[0], prob_3ch[1], prob_3ch[2]

    m = (
        (p_mask >= t_lo)
        | ((p_mask >= 0.26) & (p_spine >= spine_thr))
        | ((p_mask >= 0.28) & (p_hull >= 0.65))
    ).astype(np.uint8)
    h = (p_hull >= hull_thr).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    bridged = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
    m = np.where((bridged > 0) & (p_hull >= 0.50) & (p_mask >= 0.22), 1, m).astype(np.uint8)

    n_hull, hull_labels = cv2.connectedComponents(h, connectivity=8)
    _, base_labels = cv2.connectedComponents(m, connectivity=8)

    out = hull_labels.astype(np.int32)
    miss = (m > 0) & (out == 0)
    out[miss] = base_labels[miss] + n_hull
    out[m == 0] = 0

    instances: list[np.ndarray] = []
    values, counts = np.unique(out, return_counts=True)
    for value, count in zip(values, counts):
        if value <= 0 or count < min_area:
            continue
        comp = out == value
        vp = p_mask[comp]
        vh = p_hull[comp]
        if (float(vp.max()) >= t_hi and float(vp.mean()) >= min_meanp) or (
            float(vh.max()) >= 0.92 and float(vp.max()) >= 0.62 and count >= 380
        ):
            instances.append(comp.astype(np.uint8))
    return instances


def erode_masks(instances: list[np.ndarray], iterations: int = 1) -> list[np.ndarray]:
    """Strip outer probability halo (1 px square kernel matches polygon GT thickness)."""
    if iterations <= 0:
        return instances
    k3 = np.ones((3, 3), np.uint8)
    out: list[np.ndarray] = []
    for inst in instances:
        eroded = cv2.erode(inst.astype(np.uint8), k3, iterations=iterations)
        if int(eroded.sum()) > 0:
            out.append(eroded)
    return out


def instances_to_rows(
    stem: str,
    instances: list[np.ndarray],
    min_area: int = 65,
    size: int = 2048,
) -> list[dict[str, str]]:
    """Convert instance masks into strictly disjoint RLE rows for Kaggle evaluation."""
    occupied = np.zeros((size, size), dtype=bool)
    rows: list[dict[str, str]] = []
    idx = 1
    for mask in instances:
        m = mask.astype(bool) & ~occupied
        if int(m.sum()) < min_area:
            continue
        occupied |= m
        rows.append({"filament_id": f"{stem}_{idx}", "segmentation_rle": mask_to_rle(m)})
        idx += 1
    return rows


def candidate_features(mask_obj: Mask, prob_3ch: np.ndarray) -> list[float]:
    """Compute geometric, probability, and solar-disk radial features for FP filtering."""
    arr = mask_obj.arr
    y0, x0, h, w = mask_obj.y0, mask_obj.x0, mask_obj.h, mask_obj.w
    area = float(mask_obj.area)
    if area == 0:
        return [0.0] * len(FEATURE_NAMES)

    p_mask = prob_3ch[0, y0 : y0 + h, x0 : x0 + w][arr]
    p_hull = prob_3ch[1, y0 : y0 + h, x0 : x0 + w][arr]
    p_spine = prob_3ch[2, y0 : y0 + h, x0 : x0 + w][arr]

    dil = ndimage.binary_dilation(arr, np.ones((5, 5), bool))
    ring = dil & ~arr
    ring_mask = prob_3ch[0, y0 : y0 + h, x0 : x0 + w][ring]
    ring_hull = prob_3ch[1, y0 : y0 + h, x0 : x0 + w][ring]

    cnts, _ = cv2.findContours(arr.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = float(sum(cv2.arcLength(c, True) for c in cnts)) or 1.0
    elongation = float(perimeter**2 / (4.0 * np.pi * area + 1e-6))
    hull_area = float(sum(cv2.contourArea(cv2.convexHull(c)) for c in cnts)) or 1.0
    solidity = area / hull_area

    ys, xs = np.nonzero(arr)
    cy, cx = ys.mean() + y0, xs.mean() + x0
    rho = float(np.hypot(cy - 1023.5, cx - 1024.0) / 900.0)

    return [
        area,
        float(np.sqrt(area)),
        float(w),
        float(h),
        float(w) / max(h, 1),
        area / max(w * h, 1),
        float(p_mask.mean()),
        float(p_mask.max()),
        float(np.percentile(p_mask, 90)),
        float(p_mask.std()),
        float(p_hull.mean()),
        float(p_hull.max()),
        float(p_spine.mean()),
        float(p_spine.max()),
        float(ring_mask.mean()) if ring_mask.size else 0.0,
        float(ring_hull.mean()) if ring_hull.size else 0.0,
        elongation,
        solidity,
        rho,
    ]


def filter_candidates_with_classifier(
    instances: list[np.ndarray],
    prob_3ch: np.ndarray,
    clf,
    thr: float = 0.42,
    erode_iter: int = 1,
) -> list[np.ndarray]:
    """Score candidate masks with a trained scikit-learn classifier and apply boundary erosion."""
    cands: list[Mask] = []
    for inst in instances:
        mk = mask_from_array(inst)
        if mk is not None:
            cands.append(mk)
    if not cands:
        return []

    feats = np.array([candidate_features(mk, prob_3ch) for mk in cands], dtype=np.float32)
    scores = clf.predict_proba(feats)[:, 1]

    k3 = np.ones((3, 3), np.uint8)
    kept: list[np.ndarray] = []
    for mk, score in zip(cands, scores):
        if float(score) < thr:
            continue
        sub = mk.arr.astype(np.uint8)
        if erode_iter > 0:
            sub = cv2.erode(sub, k3, iterations=erode_iter)
        if int(sub.sum()) == 0:
            continue
        full = np.zeros((2048, 2048), dtype=np.uint8)
        full[mk.y0 : mk.y1, mk.x0 : mk.x1] = sub
        kept.append(full)
    return kept
