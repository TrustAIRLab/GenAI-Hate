"""
Run AIDE detection scoring on images from CLIP retrieval results.

Pipeline:
1. Load CLIP retrieval JSON (e.g., pajeet_images.json)
2. Run AIDE inference over all images
3. Output merged JSON (CLIP score + AIDE score + stats)

Usage:
    python aide_score_clip_results.py --input results/pajeet_images.json
    python aide_score_clip_results.py --input results/pajeet_images.json results/curry_images.json
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import numpy as np
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Paths ──
SCRIPT_DIR = Path(__file__).resolve().parent  # aide_detector/
PROJECT_ROOT = SCRIPT_DIR.parent.parent  # anti-indian-semitism/
AIDE_ROOT = SCRIPT_DIR / "AIDE"
POL_DUMPS_ROOT = PROJECT_ROOT  # images are relative to this

if str(AIDE_ROOT) not in sys.path:
    sys.path.insert(0, str(AIDE_ROOT))

from models.AIDE import AIDE as build_aide
from data.datasets import transform_before_test, transform_train
from data.dct import DCT_base_Rec_Module

DEFAULT_CHECKPOINT = str(AIDE_ROOT / "checkpoint" / "GenImage_train.pth")


# ═══════════════════════  AIDE Model  ═══════════════════════

def load_model(checkpoint, device):
    model = build_aide(resnet_path=None, convnext_path=None)
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    model.to(device).eval()
    return model


def prepare_input(image_path, dct_module, device):
    img = Image.open(image_path).convert("RGB")
    img_t = transform_before_test(img)
    x_minmin, x_maxmax, x_minmin1, x_maxmax1 = dct_module(img_t)
    x0 = transform_train(img_t)
    x_minmin = transform_train(x_minmin)
    x_maxmax = transform_train(x_maxmax)
    x_minmin1 = transform_train(x_minmin1)
    x_maxmax1 = transform_train(x_maxmax1)
    x = torch.stack([x_minmin, x_maxmax, x_minmin1, x_maxmax1, x0], dim=0).unsqueeze(0)
    return x.to(device, non_blocking=True)


@torch.no_grad()
def run_aide_inference(model, dct_module, items, device, log_interval=120):
    """
    Run AIDE on a list of (path, name) tuples.
    Returns dict: name -> {aide_score, aide_label}.
    """
    results = {}
    start = time.time()
    last_log = start

    for i, (img_path, name) in enumerate(items):
        try:
            x = prepare_input(img_path, dct_module, device)
            out = model(x)
            logit = out[0, 1].item()
            prob = 1 / (1 + math.exp(-logit))
            label = int(logit > 0)
            results[name] = {
                "aide_score": round(prob, 6),
                "aide_label": bool(label),
            }
        except Exception as e:
            print(f"  [WARN] {img_path}: {e}")
            continue

        now = time.time()
        if now - last_log >= log_interval:
            elapsed = now - start
            speed = (i + 1) / elapsed
            eta = (len(items) - i - 1) / speed if speed > 0 else 0
            print(f"  [PROGRESS] {i+1}/{len(items)}  "
                  f"speed={speed:.1f} img/s  "
                  f"ETA={eta/60:.1f}min", flush=True)
            last_log = now

    elapsed = time.time() - start
    print(f"  Inference done: {len(results)}/{len(items)} images, "
          f"{elapsed/60:.1f}min", flush=True)
    return results


# ═══════════════════════  Main Logic  ═══════════════════════

def process_one_file(input_path, model, dct_module, device, output_dir):
    """Process a single CLIP search result JSON."""
    print(f"\n{'='*60}")
    print(f"Processing: {input_path}")

    with open(input_path) as f:
        data = json.load(f)

    query = data.get("query", "unknown")
    images = data["images"]
    print(f"  Query: {query}, Total images: {len(images)}")

    # Resolve image paths and collect items for inference
    items = []  # (full_path, filename)
    skipped = 0
    for img in images:
        img_path = POL_DUMPS_ROOT / img["path"]
        if img_path.exists():
            items.append((str(img_path), Path(img["path"]).name))
        else:
            skipped += 1

    print(f"  Found {len(items)} image files on disk, {skipped} missing")

    # Run inference on all images
    scores = run_aide_inference(model, dct_module, items, device)

    # Build output
    output_images = []
    for img in images:
        name = Path(img["path"]).name
        entry = {"path": img["path"], "clip_score": img["score"]}
        if name in scores:
            entry["aide_score"] = scores[name]["aide_score"]
            entry["aide_label"] = scores[name]["aide_label"]
        else:
            entry["aide_score"] = None
            entry["aide_label"] = None
        output_images.append(entry)

    # Stats
    scores_list = [v["aide_score"] for v in scores.values()]
    stats = {}
    if scores_list:
        s = np.array(scores_list)
        stats = {
            "count": len(s),
            "mean": round(float(s.mean()), 4),
            "median": round(float(np.median(s)), 4),
            "std": round(float(s.std()), 4),
            "ai_rate_0.5": round(float((s > 0.5).mean() * 100), 2),
            "ai_rate_0.6": round(float((s > 0.6).mean() * 100), 2),
            "ai_rate_0.7": round(float((s > 0.7).mean() * 100), 2),
            "ai_rate_0.8": round(float((s > 0.8).mean() * 100), 2),
        }

    output = {
        "query": query,
        "clip_threshold": data.get("threshold"),
        "total_images": len(images),
        "aide_scored": len(scores),
        "aide_not_found": len(images) - len(scores),
        "aide_stats": stats,
        "images": output_images,
    }

    # Save
    stem = Path(input_path).stem
    out_path = Path(output_dir) / f"{stem}_aide.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    # Print summary
    print(f"\n  Summary for '{query}':")
    print(f"    Total: {len(images)}, AIDE scored: {len(scores)}, Missing: {len(images)-len(scores)}")
    if stats:
        print(f"    Score: mean={stats['mean']}, median={stats['median']}")
        print(f"    AI rate: >0.5={stats['ai_rate_0.5']}%  >0.7={stats['ai_rate_0.7']}%")

    return output


# ═══════════════════════  CLI  ═══════════════════════

def main():
    parser = argparse.ArgumentParser("AIDE scoring for CLIP search results")
    parser.add_argument("--input", type=str, nargs="+", required=True,
                        help="CLIP search result JSON file(s)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--resume", type=str, default=DEFAULT_CHECKPOINT,
                        help="AIDE checkpoint path")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading AIDE model from {args.resume} ...")
    model = load_model(args.resume, device)
    dct_module = DCT_base_Rec_Module()
    print(f"Model loaded on {device}")

    for input_path in args.input:
        output_dir = args.output_dir or str(Path(input_path).parent)
        process_one_file(input_path, model, dct_module, device, output_dir)

    print(f"\nAll done!")


if __name__ == "__main__":
    main()
