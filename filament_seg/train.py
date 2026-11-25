"""Training script for EdgeAttResUNet3Head and ConvNeXtAttUNet3Head."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import FilamentPatchDataset, build_image_records, load_annotations, make_folds
from .inference import predict_full_disk
from .metrics import evaluate_pq, mask_from_array
from .model import ConvNeXtAttUNet3Head, EdgeAttResUNet3Head, multi_head_loss
from .postprocess import erode_masks, extract_instances
from .preprocess import prepare_channels


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_files: list[str],
    img_dir: Path,
    gt_groups: dict,
    device: str,
    max_images: int = 32,
    erode_iter: int = 1,
):
    model.eval()
    preds = {}
    for fn in val_files[:max_images]:
        gray = cv2.imread(str(img_dir / fn), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        x_full = prepare_channels(gray)
        probs = predict_full_disk(model, x_full, tile=2048, use_tta=True, device=device)
        insts = erode_masks(extract_instances(probs), iterations=erode_iter)
        masks = [mask_from_array(m) for m in insts]
        preds[fn] = [m for m in masks if m is not None]

    sub_gt = {k: v for k, v in gt_groups.items() if k[0] in preds}
    res = evaluate_pq(sub_gt, preds)
    model.train()
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description="Train multi-head solar filament segmentation network")
    ap.add_argument("--data-dir", required=True, help="Path to MAGFiLO_1.0_Kaggle_2026 root")
    ap.add_argument("--arch", choices=["resnet34", "convnext"], default="resnet34")
    ap.add_argument("--fold", type=int, default=0, help="Fold index 0..4, or -1 for all training images")
    ap.add_argument("--crop-size", type=int, default=640)
    ap.add_argument("--batch", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--tversky", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-every", type=int, default=4)
    ap.add_argument("--cache-dir", default=None, help="Optional directory of cached uint8 .npy channels")
    ap.add_argument("--init-from", default=None, help="Optional checkpoint for fine-tuning")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="runs/fold0")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    ann_json = data_dir / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    train_img_dir = data_dir / "train" / "train_images"

    gt_groups, meta = load_annotations(ann_json)
    records = build_image_records(ann_json)
    folds = make_folds(meta, n_splits=5)

    if args.fold < 0:
        val_files = []
        train_files = sorted(folds.index)
    else:
        val_files = sorted(f for f in folds.index if folds[f] == args.fold)
        train_files = sorted(f for f in folds.index if folds[f] != args.fold)

    train_records = [records[f] for f in train_files if f in records]
    print(f"Arch: {args.arch} | Train observations: {len(train_records)} | Val: {len(val_files)} | Device: {device}")

    ds = FilamentPatchDataset(
        train_records,
        train_img_dir,
        crop_size=args.crop_size,
        cache_dir=args.cache_dir,
        consensus_hull=(args.arch == "convnext"),
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device == "cuda"),
        drop_last=True,
        persistent_workers=args.workers > 0,
    )

    if args.arch == "convnext":
        model: torch.nn.Module = ConvNeXtAttUNet3Head(in_channels=3, pretrained=args.init_from is None)
    else:
        model = EdgeAttResUNet3Head(in_channels=3, pretrained=args.init_from is None)

    if args.init_from:
        state = torch.load(args.init_from, map_location="cpu", weights_only=True)
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_pq = -1.0
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for bx, by in loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(bx)
                loss = multi_head_loss(logits, by, tversky_w=args.tversky)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()

        scheduler.step()
        elapsed = time.time() - t0
        entry: dict = {
            "epoch": epoch,
            "train_loss": round(running_loss / max(1, len(loader)), 5),
            "seconds": round(elapsed, 1),
        }

        if val_files and (epoch % args.val_every == 0 or epoch == args.epochs):
            res = validate(model, val_files, train_img_dir, gt_groups, device)
            entry.update(res.as_dict())
            print(
                f"Epoch {epoch:02d}/{args.epochs} | loss {entry['train_loss']:.4f} | "
                f"val PQ {res.pq:.4f} (SQ {res.sq:.3f}, RQ {res.rq:.3f}) | {elapsed:.0f}s",
                flush=True,
            )
            if res.pq > best_pq:
                best_pq = res.pq
                torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            print(f"Epoch {epoch:02d}/{args.epochs} | loss {entry['train_loss']:.4f} | {elapsed:.0f}s", flush=True)

        history.append(entry)
        torch.save(model.state_dict(), out_dir / "last.pt")
        (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    if not val_files:
        shutil.copyfile(out_dir / "last.pt", out_dir / "best.pt")


if __name__ == "__main__":
    main()
