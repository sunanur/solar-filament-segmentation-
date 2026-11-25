"""Full-disk 2048x2048 inference and RLE submission CLI."""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from .model import load_model
from .postprocess import (
    erode_masks,
    extract_instances,
    filter_candidates_with_classifier,
    instances_to_rows,
)
from .preprocess import prepare_channels

__all__ = ["tile_coords", "predict_full_disk", "run_inference"]


def tile_coords(tile: int, image_size: int = 2048) -> list[tuple[int, int]]:
    if tile >= image_size:
        return [(0, 0)]
    n = 2 if tile >= 1024 else 4
    stride = (image_size - tile) / (n - 1)
    coords = sorted({int(round(i * stride)) for i in range(n)})
    return [(y, x) for y in coords for x in coords]


@torch.no_grad()
def predict_full_disk(
    model: torch.nn.Module,
    x_full: np.ndarray,
    *,
    tile: int = 2048,
    use_tta: bool = True,
    device: str = "cuda",
) -> np.ndarray:
    """Predict 3-channel probability map over a 2048x2048 frame (single pass by default)."""
    coords = tile_coords(tile, image_size=x_full.shape[-1])
    acc = np.zeros_like(x_full, dtype=np.float32)
    weight = np.zeros((1, x_full.shape[1], x_full.shape[2]), dtype=np.float32)

    for y0, x0 in coords:
        patch = torch.from_numpy(
            np.ascontiguousarray(x_full[:, y0 : y0 + tile, x0 : x0 + tile])[None]
        ).to(device, dtype=torch.float32)

        with torch.amp.autocast("cuda", enabled=device.startswith("cuda")):
            prob = torch.sigmoid(model(patch))
            if use_tta:
                prob = prob + torch.flip(torch.sigmoid(model(torch.flip(patch, dims=[3]))), dims=[3])
                prob = prob + torch.flip(torch.sigmoid(model(torch.flip(patch, dims=[2]))), dims=[2])
                prob = prob / 3.0

        acc[:, y0 : y0 + tile, x0 : x0 + tile] += prob[0].float().cpu().numpy()
        weight[:, y0 : y0 + tile, x0 : x0 + tile] += 1.0

    return acc / np.maximum(weight, 1.0)


def run_inference(
    images_dir: str | Path,
    weights: list[str],
    out_csv: str | Path,
    *,
    tile: int = 2048,
    use_tta: bool = True,
    erode_iter: int = 1,
    fp_filter_path: str | None = None,
    fp_thr: float = 0.42,
    device: str | None = None,
) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = [load_model(w, device=device) for w in weights]
    clf = pickle.loads(Path(fp_filter_path).read_bytes()) if fp_filter_path else None

    paths = sorted(glob.glob(os.path.join(str(images_dir), "*.*")))
    rows: list[dict[str, str]] = []
    t0 = time.time()

    for i, path in enumerate(paths, start=1):
        stem = os.path.splitext(os.path.basename(path))[0]
        gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue

        x_full = prepare_channels(gray)
        prob_acc = np.zeros_like(x_full, dtype=np.float32)
        for m in models:
            prob_acc += predict_full_disk(m, x_full, tile=tile, use_tta=use_tta, device=device)
        prob_mean = prob_acc / len(models)

        instances = extract_instances(prob_mean)
        if clf is not None:
            instances = filter_candidates_with_classifier(
                instances, prob_mean, clf, thr=fp_thr, erode_iter=erode_iter
            )
        elif erode_iter > 0:
            instances = erode_masks(instances, iterations=erode_iter)

        rows.extend(instances_to_rows(stem, instances))
        if i % 15 == 0 or i == len(paths):
            rate = (time.time() - t0) / i
            print(f"[{i}/{len(paths)}] {stem}: {len(instances)} instances ({rate:.2f} s/img)", flush=True)

    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv} ({len(df)} instances)", flush=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Run full-disk solar filament segmentation inference")
    ap.add_argument("--images", required=True, help="Directory of test H-alpha images")
    ap.add_argument("--weights", nargs="+", required=True, help="One or more checkpoint paths (.pt)")
    ap.add_argument("--out", default="outputs/submission.csv", help="Output RLE CSV path")
    ap.add_argument("--tile", type=int, default=2048, help="Inference window size (2048 = single pass)")
    ap.add_argument("--erode", type=int, default=1, help="Morphological boundary erosion iterations")
    ap.add_argument("--fp-filter", default=None, help="Optional pickled candidate classifier")
    ap.add_argument("--fp-thr", type=float, default=0.42, help="Classifier probability threshold")
    ap.add_argument("--no-tta", action="store_true", help="Disable 3x flip TTA")
    args = ap.parse_args()

    run_inference(
        args.images,
        args.weights,
        args.out,
        tile=args.tile,
        use_tta=not args.no_tta,
        erode_iter=args.erode,
        fp_filter_path=args.fp_filter,
        fp_thr=args.fp_thr,
    )


if __name__ == "__main__":
    main()
