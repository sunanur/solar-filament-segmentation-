"""Full-disk GONG H-alpha channel preparation (2048x2048 -> 3 channels)."""

from __future__ import annotations

import cv2
import numpy as np

__all__ = ["prepare_channels"]

_CLAHE = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(16, 16))


def prepare_channels(gray: np.ndarray, size: int = 2048) -> np.ndarray:
    """Build a 3-channel float32 tensor in [0, 1] from a grayscale H-alpha frame.

    Channels:
        0: Percentile-clipped solar disk intensity (off-disk zeroed)
        1: Local contrast enhanced frame via CLAHE (16x16 tiles)
        2: Normalized Sobel gradient magnitude
    """
    h, w = gray.shape[:2]
    if h != size or w != size:
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)

    sub = gray[::4, ::4]
    sh, sw = sub.shape
    yy, xx = np.ogrid[:sh, :sw]
    disk_sub = ((xx - sw // 2) ** 2 + (yy - sh // 2) ** 2) < int(0.453 * sw) ** 2

    yy_f, xx_f = np.ogrid[:size, :size]
    disk = ((xx_f - size // 2) ** 2 + (yy_f - size // 2) ** 2) < int(0.454 * size) ** 2

    if np.any(disk_sub):
        lo, hi = np.percentile(sub[disk_sub], (1, 99))
        norm = np.clip((gray.astype(np.float32) - float(lo)) / (float(hi - lo) + 1e-6), 0.0, 1.0)
        norm[~disk] = 0.0
    else:
        norm = gray.astype(np.float32) / 255.0

    u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    cl = _CLAHE.apply(u8).astype(np.float32) / 255.0

    dx = cv2.Sobel(u8, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(u8, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(dx * dx + dy * dy)
    p99 = float(np.percentile(mag[::4, ::4], 99))
    mag = np.clip(mag / (p99 + 1e-6), 0.0, 1.0)

    return np.stack([norm, cl, mag], axis=0).astype(np.float32)
