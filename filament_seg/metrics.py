"""Bounding-box cropped binary masks and pooled multi-annotator Panoptic Quality."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from PIL import Image, ImageDraw

__all__ = [
    "Mask",
    "EvalResult",
    "mask_from_array",
    "mask_from_polygon",
    "iou",
    "iou_matrix",
    "evaluate_pq",
]

IOU_THRESHOLD = 0.5


@dataclass(frozen=True, eq=False)
class Mask:
    """Cropped binary mask stored inside its tight bounding box to save memory at 2048x2048."""

    y0: int
    x0: int
    arr: np.ndarray
    area: int

    @property
    def h(self) -> int:
        return int(self.arr.shape[0])

    @property
    def w(self) -> int:
        return int(self.arr.shape[1])

    @property
    def y1(self) -> int:
        return self.y0 + self.h

    @property
    def x1(self) -> int:
        return self.x0 + self.w

    def to_full(self, height: int = 2048, width: int = 2048) -> np.ndarray:
        full = np.zeros((height, width), dtype=bool)
        full[self.y0 : self.y1, self.x0 : self.x1] = self.arr
        return full


def mask_from_array(mask: np.ndarray, y_offset: int = 0, x_offset: int = 0) -> Mask | None:
    arr = np.asarray(mask, dtype=bool)
    if not arr.any():
        return None
    rows = np.flatnonzero(arr.any(axis=1))
    cols = np.flatnonzero(arr.any(axis=0))
    r0, r1 = int(rows[0]), int(rows[-1]) + 1
    c0, c1 = int(cols[0]), int(cols[-1]) + 1
    sub = np.ascontiguousarray(arr[r0:r1, c0:c1])
    return Mask(y_offset + r0, x_offset + c0, sub, int(sub.sum()))


def mask_from_polygon(
    segmentation: list,
    height: int = 2048,
    width: int = 2048,
    supersample: int = 4,
) -> Mask | None:
    """Rasterize COCO polygon rings using supersampled coverage."""
    if not segmentation:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for ring in segmentation:
        xs.extend(ring[0::2])
        ys.extend(ring[1::2])
    if not xs:
        return None

    x_lo = max(0, int(np.floor(min(xs))) - 2)
    x_hi = min(width, int(np.ceil(max(xs))) + 2)
    y_lo = max(0, int(np.floor(min(ys))) - 2)
    y_hi = min(height, int(np.ceil(max(ys))) + 2)
    if x_hi <= x_lo or y_hi <= y_lo:
        return None

    s = max(1, int(supersample))
    bw, bh = x_hi - x_lo, y_hi - y_lo
    canvas = Image.new("1", (bw * s, bh * s), 0)
    draw = ImageDraw.Draw(canvas)
    for ring in segmentation:
        pts = [
            ((ring[i] - x_lo) * s, (ring[i + 1] - y_lo) * s)
            for i in range(0, len(ring) - 1, 2)
        ]
        if len(pts) >= 3:
            draw.polygon(pts, fill=1)

    hi = np.array(canvas, dtype=np.float32)
    cov = hi.reshape(bh, s, bw, s).mean(axis=(1, 3)) if s > 1 else hi
    return mask_from_array(cov >= 0.5, y_lo, x_lo)


def iou(a: Mask, b: Mask) -> float:
    oy0, ox0 = max(a.y0, b.y0), max(a.x0, b.x0)
    oy1, ox1 = min(a.y1, b.y1), min(a.x1, b.x1)
    if oy1 <= oy0 or ox1 <= ox0:
        return 0.0

    sub_a = a.arr[oy0 - a.y0 : oy1 - a.y0, ox0 - a.x0 : ox1 - a.x0]
    sub_b = b.arr[oy0 - b.y0 : oy1 - b.y0, ox0 - b.x0 : ox1 - b.x0]
    inter = int(np.count_nonzero(sub_a & sub_b))
    if inter == 0:
        return 0.0
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(gt: list[Mask], pred: list[Mask]) -> np.ndarray:
    out = np.zeros((len(gt), len(pred)), dtype=np.float64)
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            out[i, j] = iou(g, p)
    return out


@dataclass
class EvalResult:
    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    n_units: int
    iou_scores: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0))

    def as_dict(self) -> dict:
        return {
            "pq": round(self.pq, 4),
            "sq": round(self.sq, 4),
            "rq": round(self.rq, 4),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "n_units": self.n_units,
        }


def evaluate_pq(
    gt_groups: dict[tuple[str, str], list[Mask]],
    pred_by_image: dict[str, list[Mask]],
    iou_thr: float = IOU_THRESHOLD,
) -> EvalResult:
    """Compute pooled Panoptic Quality across all (file_name, annotator_id) pairs.

    In MAGFiLO, a single observation annotated by K solar physicists contributes
    K evaluation units to the global TP/FP/FN pool.
    """
    tp_total = 0
    fp_total = 0
    fn_total = 0
    iou_sum = 0.0
    n_units = 0
    unit_ious: list[float] = []

    for (file_name, _), gt_masks in gt_groups.items():
        pred_masks = pred_by_image.get(file_name, [])
        n_gt, n_pred = len(gt_masks), len(pred_masks)
        if n_gt == 0 and n_pred == 0:
            continue
        n_units += 1
        if n_gt == 0:
            fp_total += n_pred
            continue
        if n_pred == 0:
            fn_total += n_gt
            continue

        mat = iou_matrix(gt_masks, pred_masks)
        hit = mat > iou_thr
        tp = int(hit.sum())
        hits = mat[hit]
        tp_total += tp
        fp_total += int((hit.sum(axis=0) == 0).sum())
        fn_total += int((hit.sum(axis=1) == 0).sum())
        iou_sum += float(hits.sum())
        if tp > 0:
            unit_ious.append(float(hits.mean()))

    denom = tp_total + 0.5 * fp_total + 0.5 * fn_total
    pq = iou_sum / denom if denom > 0 else 0.0
    sq = iou_sum / tp_total if tp_total > 0 else 0.0
    rq = tp_total / denom if denom > 0 else 0.0
    return EvalResult(
        pq=pq,
        sq=sq,
        rq=rq,
        tp=tp_total,
        fp=fp_total,
        fn=fn_total,
        n_units=n_units,
        iou_scores=np.array(unit_ious, dtype=float),
    )
