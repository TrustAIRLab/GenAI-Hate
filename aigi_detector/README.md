# AI-Generated Image Detection (aigi_detector)

Score CLIP-retrieved 4chan /pol/ images with the AIDE detector to measure AI-generated image prevalence over time.

## Script

`aide_detector/aide_score_clip_results.py` — runs AIDE inference on images retrieved by `clip_search` and outputs per-image scores.

## Usage

```bash
python aigi_detector/aide_detector/aide_score_clip_results.py \
  --input clip_search/results/pajeet_images.json \
  --output-dir aigi_detector/aide_detector/results \
  --device cuda
```

Requires the AIDE model checkpoint. See the original repository for installation:
https://github.com/shilinyan99/AIDE
