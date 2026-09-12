#!/usr/bin/env python3
\"\"\"
Enhanced NOAA MBES ShipwreckFinder ONNX Inference Pipeline
Computes dual-channel normalized depth + 315°/45° hillshade relief
and performs accelerated ONNX inference.
\"\"\"

import argparse
import json
import os
import numpy as np
import onnxruntime as ort

def compute_hillshade(elevation, az_deg=315.0, alt_deg=45.0):
    az_rad = np.radians(360.0 - az_deg + 90.0)
    alt_rad = np.radians(alt_deg)
    
    dy, dx = np.gradient(elevation)
    slope = np.pi / 2.0 - np.arctan(np.sqrt(dx * dx + dy * dy))
    aspect = np.arctan2(-dx, dy)
    
    shaded = np.sin(alt_rad) * np.sin(slope) + np.cos(alt_rad) * np.cos(slope) * np.cos(az_rad - aspect)
    return np.clip(shaded, 0.0, 1.0).astype(np.float32)

def prepare_chip(patch, target_size=416):
    h, w = patch.shape
    if h != target_size or w != target_size:
        pad_h = max(0, target_size - h)
        pad_w = max(0, target_size - w)
        patch = np.pad(patch[:target_size, :target_size], ((0, pad_h), (0, pad_w)), mode=\"edge\")
        
    patch = np.nan_to_num(patch, nan=float(np.nanmean(patch)) if np.isnan(patch).any() else 0.0)
    p_min, p_max = patch.min(), patch.max()
    norm_depth = (patch - p_min) / (p_max - p_min) if p_max > p_min else np.zeros_like(patch)
    hillshade = compute_hillshade(patch, az_deg=315.0, alt_deg=45.0)
    
    # [1, 2, 416, 416]
    return np.stack([norm_depth, hillshade], axis=0)[np.newaxis, ...].astype(np.float32)

def main():
    parser = argparse.ArgumentParser(description=\"Run Enhanced ONNX ShipwreckFinder\")
    parser.add_argument(\"--model\", default=\"models/unet_hillshade.onnx\", help=\"Path to ONNX weights\")
    parser.add_argument(\"--npy\", help=\"Path to 2D numpy elevation array chip\")
    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f\"Model not found at {args.model}. Download weights from release assets.\")
        return

    session = ort.InferenceSession(args.model, providers=[\"CPUExecutionProvider\"])
    print(f\"Loaded ONNX session: {args.model}\")
    
    if args.npy and os.path.exists(args.npy):
        patch = np.load(args.npy)
        tensor = prepare_chip(patch)
        input_name = session.get_inputs()[0].name
        outs = session.run(None, {input_name: tensor})
        logits = outs[0][0]
        exp_logits = np.exp(logits - np.max(logits, axis=0, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
        ship_prob = probs[1]
        print(f\"Max Shipwreck Probability: {float(np.max(ship_prob)):.4f}\")
        print(f\"Active Shipwreck Pixels (P > 0.5): {int(np.sum(ship_prob > 0.5))}\")

if __name__ == \"__main__\":
    main()
