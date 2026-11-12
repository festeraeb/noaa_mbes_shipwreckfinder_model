#!/usr/bin/env python3
"""
Enhanced NOAA MBES ShipwreckFinder ONNX Inference Pipeline
Computes dual-channel normalized depth + 315°/45° hillshade relief
and performs accelerated ONNX inference.
"""

import argparse
import json
import os
import numpy as np

def compute_hillshade(elevation, az_deg=315.0, alt_deg=45.0, cell_size=1.0):
    """Hillshade in [0, 1], the same computation as MBESDataset.compute_hillshade
    (util/data.py): central differences on an edge-padded array, the same aspect
    convention, and the same 8-bit quantisation the stored training hillshades have."""
    az_rad = np.radians(360.0 - az_deg + 90.0)
    alt_rad = np.radians(alt_deg)

    padded = np.pad(elevation, pad_width=1, mode="edge")
    dzdx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / (2 * cell_size)
    dzdy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / (2 * cell_size)

    slope = np.arctan(np.hypot(dzdx, dzdy))
    aspect = np.arctan2(dzdy, -dzdx)
    aspect = np.where(aspect < 0, 2 * np.pi + aspect, aspect)

    shaded = np.sin(alt_rad) * np.cos(slope) + np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    return (np.clip(shaded, 0, 1) * 255).astype(np.uint8).astype(np.float32) / 255.0


def resize_nearest(image, size):
    """Resample the whole chip to size x size, nearest neighbour, as the training
    loader does (transforms.Resize(..., NEAREST) in util/data.py)."""
    h, w = image.shape
    rows = np.minimum((np.arange(size) * (h / size)).astype(np.int64), h - 1)
    cols = np.minimum((np.arange(size) * (w / size)).astype(np.int64), w - 1)
    return image[rows][:, cols]


def normalize_nonzero(image):
    """Scale the nonzero (valid) depths to [0, 1] and leave zeros (no data) at zero,
    as util/utils.py normalize_nonzero does for the training loader."""
    image = image.astype(np.float32).copy()
    valid = image != 0
    if valid.any():
        lo, hi = image[valid].min(), image[valid].max()
        if hi > lo:
            if not (lo >= 0 and hi <= 1):
                image[valid] = (image[valid] - lo) / (hi - lo)
        else:
            # one depth only: no relief to scale (the training code would divide by zero)
            image[valid] = 0.0
    return image


def prepare_chip(patch, target_size=416):
    """Build the [1, 2, size, size] model input from a 2D elevation chip.

    Missing samples may be NaN/inf or 0 (the training data's no-data value).
    Raises ValueError when the chip has no usable depth."""
    patch = np.asarray(patch, dtype=np.float32)
    if patch.ndim != 2 or patch.size == 0:
        raise ValueError(f"expected a 2D elevation array, got shape {patch.shape}")
    valid = np.isfinite(patch) & (patch != 0)
    if not valid.any():
        raise ValueError("the chip has no finite, nonzero depth samples")

    # depth channel: no data is 0, the whole chip is resampled, valid depths span 0..1
    depth = np.where(valid, patch, 0.0).astype(np.float32)
    norm_depth = normalize_nonzero(resize_nearest(depth, target_size))

    # hillshade channel: computed at the chip's own resolution, gaps filled with the
    # mean of the valid depths so a gap does not read as a cliff, then resampled
    filled = np.where(valid, patch, float(patch[valid].mean())).astype(np.float32)
    hillshade = resize_nearest(compute_hillshade(filled, az_deg=315.0, alt_deg=45.0), target_size)

    return np.stack([norm_depth, hillshade], axis=0)[np.newaxis, ...].astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description="Run Enhanced ONNX ShipwreckFinder")
    parser.add_argument("--model", default="models/unet_hillshade.onnx", help="Path to ONNX weights")
    parser.add_argument("--npy", help="Path to 2D numpy elevation array chip")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f"Model not found at {args.model}. Download weights from release assets.")
        return 1

    import onnxruntime as ort  # imported here so the preprocessing can be tested without it

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    print(f"Loaded ONNX session: {args.model}")
    
    if args.npy and os.path.exists(args.npy):
        patch = np.load(args.npy)
        try:
            tensor = prepare_chip(patch)
        except ValueError as e:
            print(f"Unusable chip {args.npy}: {e}")
            return 2
        input_name = session.get_inputs()[0].name
        outs = session.run(None, {input_name: tensor})
        logits = outs[0][0]
        exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        ship_prob = probs[1]
        print(f"Max Shipwreck Probability: {float(np.max(ship_prob)):.4f}")
        print(f"Active Shipwreck Pixels (P > 0.5): {int(np.sum(ship_prob > 0.5))}")

if __name__ == "__main__":
    raise SystemExit(main() or 0)
