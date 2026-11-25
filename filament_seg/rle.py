"""Standalone COCO RLE string encoder/decoder (compatible with pycocotools maskApi.c)."""

from __future__ import annotations

import numpy as np

__all__ = [
    "mask_to_counts",
    "counts_to_mask",
    "counts_to_string",
    "string_to_counts",
    "mask_to_rle",
    "rle_to_mask",
]


def mask_to_counts(mask: np.ndarray) -> list[int]:
    flat = np.asarray(mask, dtype=np.uint8).flatten(order="F")
    if flat.size == 0:
        return [0]

    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    bounds = np.concatenate(([0], changes, [flat.size]))
    lengths = np.diff(bounds).astype(np.int64).tolist()

    counts: list[int] = [0] if flat[0] == 1 else []
    counts.extend(lengths)
    return counts


def counts_to_mask(counts: list[int], height: int, width: int) -> np.ndarray:
    total = int(height) * int(width)
    flat = np.zeros(total, dtype=np.uint8)
    pos = 0
    val = 0
    for length in counts:
        length = int(length)
        if length < 0 or pos + length > total:
            raise ValueError(f"Invalid RLE run length {length} at offset {pos}")
        if val:
            flat[pos : pos + length] = 1
        pos += length
        val ^= 1
    if pos != total:
        raise ValueError(f"RLE sum {pos} does not match {total}")
    return flat.reshape((height, width), order="F").astype(bool)


def counts_to_string(counts: list[int]) -> str:
    out: list[str] = []
    for i, raw in enumerate(counts):
        x = int(raw)
        if i > 2:
            x -= int(counts[i - 2])
        more = True
        while more:
            c = x & 0x1F
            x >>= 5
            more = (x != -1) if (c & 0x10) else (x != 0)
            if more:
                c |= 0x20
            out.append(chr(c + 48))
    return "".join(out)


def string_to_counts(s: str) -> list[int]:
    counts: list[int] = []
    p = 0
    n = len(s)
    while p < n:
        x = 0
        k = 0
        more = True
        while more:
            if p >= n:
                raise ValueError("Truncated RLE string")
            c = ord(s[p]) - 48
            p += 1
            x |= (c & 0x1F) << (5 * k)
            more = bool(c & 0x20)
            k += 1
        if x & (1 << (5 * k - 1)):
            x -= 1 << (5 * k)
        if len(counts) > 2:
            x += counts[-2]
        counts.append(x)
    return counts


def mask_to_rle(mask: np.ndarray) -> str:
    return counts_to_string(mask_to_counts(mask))


def rle_to_mask(s: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    return counts_to_mask(string_to_counts(s), height, width)
