# CLIP Image Search

CLIP ViT-B/32 text-to-image semantic search over 4chan /pol/ images to detect anti-Indian and anti-Semitic visual content.

## Pipeline

1. Compute CLIP embeddings for all images → 512-d normalized vectors
2. Build a FAISS IVFPQ index for fast approximate nearest-neighbour search
3. Search the index with text queries (hate-content categories)
4. Aggregate results into weekly temporal trends

## Directory Structure

```
clip_search/
├── scripts/
│   ├── clip_embedding_compute.py  # Step 1: compute CLIP embeddings
│   └── search_images.py           # Step 2: text→image retrieval
└── configs/
    └── example_queries.txt        # Predefined query list
```

## Usage

```bash
# Compute embeddings (requires local image data)
python scripts/clip_embedding_compute.py --data-dir <image_dir> --output-dir data

# Search a single query
python scripts/search_images.py --index-dir data --query "pajeet meme"
```

## Requirements

- GPU with 12 GB+ VRAM recommended for embedding computation
- Raw 4chan /pol/ image files (not distributed; available via 4plebs archive)

## Predefined Queries

See `configs/example_queries.txt`. Categories include anti-Indian slurs/stereotypes, anti-Semitic slurs/stereotypes, and general hate symbols.
