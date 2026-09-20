"""
Text-to-Image Search using CLIP and FAISS

Search for images using text queries based on CLIP embeddings.

Usage:
    python search_images.py --query "anti-indian meme pajeet" --top-k 10
"""

import os
import argparse
import json
import numpy as np
import torch
import clip
import faiss
from pathlib import Path
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImageSearchEngine:
    """CLIP-based image search engine"""

    def __init__(self, index_dir, device='cuda', model_name='ViT-B/32'):
        """
        Initialize search engine

        Args:
            index_dir: Directory containing FAISS index and metadata
            device: 'cuda' or 'cpu'
            model_name: CLIP model name (must match the one used for embeddings)
        """
        self.index_dir = index_dir
        self.device = device
        self.model_name = model_name

        # Load CLIP model
        logger.info(f"Loading CLIP model: {model_name}")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()

        # Load metadata
        metadata_file = os.path.join(index_dir, 'index_metadata.json')
        if not os.path.exists(metadata_file):
            metadata_file = os.path.join(index_dir, 'metadata.json')

        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)

        # Load FAISS index
        index_files = list(Path(index_dir).glob('faiss_index_*.index'))
        if not index_files:
            raise FileNotFoundError(f"No FAISS index found in {index_dir}")

        index_file = str(index_files[0])
        logger.info(f"Loading FAISS index from {index_file}")

        # Load index - try direct loading first (works if you have enough RAM)
        index_size_gb = os.path.getsize(index_file) / (1024**3)
        logger.info(f"Index file size: {index_size_gb:.2f} GB")

        try:
            # Try loading directly into RAM first (fastest if you have enough memory)
            logger.info("Loading index directly into memory...")
            self.index = faiss.read_index(index_file)
            logger.info(f"Index loaded successfully into RAM")

            # If it's an IVF index, set nprobe for search quality/speed tradeoff
            if hasattr(self.index, 'nprobe'):
                # For 10M images with nlist=16384, nprobe=32-64 is a good balance
                self.index.nprobe = 32
                logger.info(f"Set nprobe={self.index.nprobe} for search")

        except (RuntimeError, MemoryError) as e:
            if "mmap" in str(e).lower() or "allocate" in str(e).lower():
                logger.error(f"Failed to load index of size {index_size_gb:.2f} GB")
                logger.error("Possible solutions:")
                logger.error("1. Rebuild the index using IVFPQ instead of IVF (more memory efficient)")
                logger.error("2. Use a machine with more RAM")
                logger.error("3. Build a smaller index with a subset of images")
                raise RuntimeError(f"Insufficient memory to load index of size {index_size_gb:.2f} GB") from e
            else:
                raise

        # Note: Cannot move IVF indices with on-disk inverted lists to GPU
        # GPU search requires the index to be fully in memory
        if device == 'cuda' and faiss.get_num_gpus() > 0:
            if not (hasattr(self.index, 'invlists') and isinstance(self.index.invlists, faiss.OnDiskInvertedLists)):
                logger.info("Moving index to GPU...")
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
            else:
                logger.warning("Cannot move index with on-disk inverted lists to GPU. Using CPU search.")

        # Load image paths
        mapping_file = os.path.join(index_dir, 'image_paths.json')
        logger.info(f"Loading image paths from {mapping_file}")
        with open(mapping_file, 'r') as f:
            self.image_paths = json.load(f)

        logger.info(f"Search engine initialized with {len(self.image_paths):,} images")

    def encode_text(self, text_query):
        """Encode text query to embedding"""
        with torch.no_grad():
            text_tokens = clip.tokenize([text_query]).to(self.device)
            text_embedding = self.model.encode_text(text_tokens)
            text_embedding = text_embedding.cpu().numpy()

            # Normalize
            text_embedding = text_embedding / np.linalg.norm(text_embedding, axis=1, keepdims=True)

        return text_embedding.astype(np.float32)

    def search(self, text_query, top_k=10, return_scores=True):
        """
        Search for images matching text query

        Args:
            text_query: Text description
            top_k: Number of results to return
            return_scores: Whether to return similarity scores

        Returns:
            List of (image_path, score) tuples if return_scores=True,
            otherwise list of image paths
        """
        # Encode text query
        text_embedding = self.encode_text(text_query)

        # Search
        scores, indices = self.index.search(text_embedding, top_k)

        # Get results
        results = []
        for i, (idx, score) in enumerate(zip(indices[0], scores[0])):
            if idx < len(self.image_paths):
                img_path = self.image_paths[idx]
                if return_scores:
                    results.append((img_path, float(score)))
                else:
                    results.append(img_path)

        return results

    def batch_search(self, text_queries, top_k=10):
        """
        Search for multiple queries

        Args:
            text_queries: List of text descriptions
            top_k: Number of results per query

        Returns:
            List of result lists
        """
        # Encode all queries
        text_embeddings = []
        for query in text_queries:
            embedding = self.encode_text(query)
            text_embeddings.append(embedding)

        text_embeddings = np.vstack(text_embeddings)

        # Search
        scores, indices = self.index.search(text_embeddings, top_k)

        # Get results
        all_results = []
        for query_idx in range(len(text_queries)):
            results = []
            for i in range(top_k):
                idx = indices[query_idx][i]
                score = scores[query_idx][i]
                if idx < len(self.image_paths):
                    results.append((self.image_paths[idx], float(score)))
            all_results.append(results)

        return all_results


def save_results(results, query, output_file):
    """Save search results to JSON file"""
    output = {
        'query': query,
        'timestamp': datetime.now().isoformat(),
        'num_results': len(results),
        'results': [
            {
                'rank': i + 1,
                'image_path': path,
                'similarity_score': score
            }
            for i, (path, score) in enumerate(results)
        ]
    }

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    logger.info(f"Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Search images using text queries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic search
  python search_images.py --index-dir embeddings --query "anti-indian meme" --top-k 20

  # Search for anti-Semitic content
  python search_images.py --index-dir embeddings --query "antisemitic caricature" --top-k 50

  # Batch search from file
  python search_images.py --index-dir embeddings --query-file queries.txt --top-k 10

  # Save results to JSON
  python search_images.py --index-dir embeddings --query "pajeet meme" --output results.json
        """
    )

    parser.add_argument(
        '--index-dir',
        type=str,
        required=True,
        help='Directory containing FAISS index and metadata'
    )

    parser.add_argument(
        '--query',
        type=str,
        default=None,
        help='Text query for search'
    )

    parser.add_argument(
        '--query-file',
        type=str,
        default=None,
        help='File containing multiple queries (one per line)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of results to return (default: 10)'
    )

    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'],
        help='Device to use (default: cuda if available)'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='ViT-B/32',
        help='CLIP model name (must match embedding computation)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file for results'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='search_results',
        help='Output directory for batch search results'
    )

    args = parser.parse_args()

    if not args.query and not args.query_file:
        parser.error("Either --query or --query-file must be provided")

    # Initialize search engine
    logger.info("=" * 60)
    logger.info("CLIP Image Search Engine")
    logger.info("=" * 60)

    engine = ImageSearchEngine(
        index_dir=args.index_dir,
        device=args.device,
        model_name=args.model
    )

    # Single query
    if args.query:
        logger.info(f"Query: {args.query}")
        logger.info(f"Searching for top {args.top_k} results...")

        results = engine.search(args.query, top_k=args.top_k, return_scores=True)

        logger.info("=" * 60)
        logger.info(f"Top {len(results)} results:")
        logger.info("=" * 60)

        for i, (path, score) in enumerate(results, 1):
            logger.info(f"{i:3d}. [Score: {score:.4f}] {path}")

        # Save results if output file specified
        if args.output:
            save_results(results, args.query, args.output)

    # Batch queries
    elif args.query_file:
        with open(args.query_file, 'r') as f:
            queries = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(queries)} queries from {args.query_file}")

        os.makedirs(args.output_dir, exist_ok=True)

        for i, query in enumerate(queries, 1):
            logger.info(f"\nProcessing query {i}/{len(queries)}: {query}")

            results = engine.search(query, top_k=args.top_k, return_scores=True)

            # Save results
            output_file = os.path.join(args.output_dir, f"query_{i:04d}.json")
            save_results(results, query, output_file)

            logger.info(f"Top 5 results:")
            for j, (path, score) in enumerate(results[:5], 1):
                logger.info(f"  {j}. [Score: {score:.4f}] {path}")

        logger.info(f"\n✅ Batch search completed. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
