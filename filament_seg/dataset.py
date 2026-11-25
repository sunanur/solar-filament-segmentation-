"""COCO annotation loading, leak-free observation splits, and patch dataset."""

from __future__ import annotations

import json
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .metrics import Mask, mask_from_array, mask_from_polygon
from .preprocess import prepare_channels
from .rle import counts_to_mask, string_to_counts

__all__ = [
    "load_annotations",
    "load_submission",
    "make_folds",
    "build_image_records",
    "FilamentPatchDataset",
]

IMAGE_SIZE = 2048


def load_annotations(json_path: str | Path) -> tuple[dict[tuple[str, str], list[Mask]], pd.DataFrame]:
    """Parse MAGFiLO COCO annotations into (file_name, image_id) mask groups."""
    with Path(json_path).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    images = {im["id"]: im for im in raw["images"]}
    gt_groups: dict[tuple[str, str], list[Mask]] = {}

    for ann in raw["annotations"]:
        image_id = ann["image_id"]
        image = images[image_id]
        file_name = image["file_name"]
        mask = mask_from_polygon(
            ann["segmentation"],
            height=image.get("height", IMAGE_SIZE),
            width=image.get("width", IMAGE_SIZE),
        )
        if mask is not None:
            gt_groups.setdefault((file_name, image_id), []).append(mask)

    rows = [
        {
            "file_name": fn,
            "annotator_image": ann_img,
            "annotator": str(ann_img).split("-", 1)[0],
            "n_masks": len(masks),
        }
        for (fn, ann_img), masks in gt_groups.items()
    ]
    meta = pd.DataFrame(rows).sort_values(["file_name", "annotator_image"]).reset_index(drop=True)
    return gt_groups, meta


def load_submission(csv_path: str | Path, image_size: int = IMAGE_SIZE) -> dict[str, list[Mask]]:
    """Decode a Kaggle RLE submission file back into cropped Mask instances."""
    df = pd.read_csv(csv_path)
    grouped: dict[str, list[Mask]] = {}
    for filament_id, rle in zip(df["filament_id"], df["segmentation_rle"]):
        stem = str(filament_id).rsplit("_", 1)[0]
        if isinstance(rle, str) and rle.strip():
            arr = counts_to_mask(string_to_counts(rle), image_size, image_size)
            mk = mask_from_array(arr)
            if mk is not None:
                grouped.setdefault(stem, []).append(mk)
    return grouped


def make_folds(meta: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.Series:
    """Stratify folds strictly by physical observation (file_name).

    Splitting on COCO image_id leaks observations across train/val because
    multiple annotators label the same physical H-alpha image under distinct IDs.
    """
    per_file = (
        meta.groupby("file_name")["annotator_image"]
        .nunique()
        .rename("n_annotators")
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    per_file["_rand"] = rng.random(len(per_file))

    folds: dict[str, int] = {}
    for _, group in per_file.groupby("n_annotators", sort=True):
        group = group.sort_values("_rand")
        offset = len(folds) % n_splits
        for i, fn in enumerate(group["file_name"]):
            folds[fn] = (offset + i) % n_splits

    return pd.Series(folds, name="fold")


def build_image_records(json_path: str | Path) -> dict[str, dict]:
    """Group raw polygon/spine annotations and centroid anchors per file_name."""
    with Path(json_path).open("r", encoding="utf-8") as fh:
        raw = json.load(fh)

    id_to_file = {im["id"]: im["file_name"] for im in raw["images"]}
    records: dict[str, dict] = {}
    for ann in raw["annotations"]:
        fn = id_to_file.get(ann["image_id"])
        if fn is None:
            continue
        rec = records.setdefault(fn, {"file_name": fn, "annotators": {}, "centers": []})
        rec["annotators"].setdefault(ann["image_id"], []).append(ann)
        for poly in ann.get("segmentation", []) or []:
            if len(poly) >= 6:
                pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                rec["centers"].append((int(pts[:, 1].mean()), int(pts[:, 0].mean())))
    return records


class FilamentPatchDataset(Dataset):
    """Random full-resolution patches with 3-head targets (soft mask, hull, spine)."""

    def __init__(
        self,
        records: list[dict],
        img_dir: str | Path,
        crop_size: int = 640,
        crops_per_image: int = 1,
        cache_dir: str | Path | None = None,
        consensus_hull: bool = False,
    ):
        self.records = records
        self.img_dir = Path(img_dir)
        self.crop_size = crop_size
        self.crops_per_image = crops_per_image
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.consensus_hull = consensus_hull
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    def __len__(self) -> int:
        return len(self.records) * self.crops_per_image

    def _load_uint8(self, file_name: str) -> np.ndarray:
        if self.cache_dir is not None:
            cached = self.cache_dir / f"{Path(file_name).stem}.npy"
            if cached.exists():
                return np.load(cached, mmap_mode="r")
        gray = cv2.imread(str(self.img_dir / file_name), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            gray = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
        ch = prepare_channels(gray, size=IMAGE_SIZE)
        return np.clip(ch * 255.0, 0, 255).astype(np.uint8)

    def _rasterize_targets(self, rec: dict, y0: int, x0: int, cs: int) -> np.ndarray:
        n_ann = max(1, len(rec["annotators"]))
        mask_sum = np.zeros((cs, cs), dtype=np.float32)
        spine_sum = np.zeros((cs, cs), dtype=np.float32)

        for anns in rec["annotators"].values():
            mk = np.zeros((cs, cs), dtype=np.uint8)
            sk = np.zeros((cs, cs), dtype=np.uint8)
            for a in anns:
                for poly in a.get("segmentation", []) or []:
                    if len(poly) >= 6:
                        pts = np.array(poly, dtype=np.float32).reshape(-1, 2) - [x0, y0]
                        cv2.fillPoly(mk, [pts.astype(np.int32)], 1)
                sp = a.get("spine", [])
                if isinstance(sp, list) and len(sp) >= 4:
                    spts = np.array(sp, dtype=np.float32).reshape(-1, 2) - [x0, y0]
                    cv2.polylines(sk, [spts.astype(np.int32)], False, 1, 3)
            mask_sum += mk
            spine_sum += sk

        soft = np.clip(mask_sum / n_ann, 0.0, 1.0).astype(np.float32)
        if self.consensus_hull and n_ann >= 3:
            union = (mask_sum >= max(1.0, 0.34 * n_ann)).astype(np.uint8)
        else:
            union = (mask_sum > 0).astype(np.uint8)
        hull = cv2.morphologyEx(union, cv2.MORPH_CLOSE, self.kernel_close).astype(np.float32)
        spine = np.clip(spine_sum / n_ann, 0.0, 1.0).astype(np.float32)
        return np.stack([soft, hull, spine], axis=0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx // self.crops_per_image]
        x_full = self._load_uint8(rec["file_name"])

        cs = self.crop_size
        centers = rec.get("centers", [])
        if centers and random.random() < 0.80:
            cy, cx = random.choice(centers)
            cy += random.randint(-140, 140)
            cx += random.randint(-140, 140)
            y0 = int(np.clip(cy - cs // 2, 0, IMAGE_SIZE - cs))
            x0 = int(np.clip(cx - cs // 2, 0, IMAGE_SIZE - cs))
        else:
            y0 = random.randint(0, IMAGE_SIZE - cs)
            x0 = random.randint(0, IMAGE_SIZE - cs)

        x = np.ascontiguousarray(x_full[:, y0 : y0 + cs, x0 : x0 + cs], dtype=np.float32) / 255.0
        y = self._rasterize_targets(rec, y0, x0, cs)

        # Dihedral D4 augmentations
        k = random.randint(0, 3)
        if k:
            x = np.rot90(x, k, axes=(1, 2)).copy()
            y = np.rot90(y, k, axes=(1, 2)).copy()
        if random.random() < 0.5:
            x = np.flip(x, axis=2).copy()
            y = np.flip(y, axis=2).copy()
        if random.random() < 0.5:
            x = np.flip(x, axis=1).copy()
            y = np.flip(y, axis=1).copy()

        return torch.from_numpy(x), torch.from_numpy(y)
