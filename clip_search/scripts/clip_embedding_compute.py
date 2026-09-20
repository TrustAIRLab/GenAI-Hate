"""
CLIP Embedding Computation for Large-Scale Image Dataset

Supports GPU acceleration, batch processing, and resume from checkpoint.

Usage:
    python clip_embedding_compute.py --data-dir pol_dumps --output-dir clip_embeddings --batch-size 256
"""

import os
import argparse
import json
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import clip
from torch.utils.data import Dataset, DataLoader
import h5py
from datetime import datetime
import logging

# Setup logging
_log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs')
os.makedirs(_log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(_log_dir, 'clip_embedding.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImagePathDataset(Dataset):
    """Dataset for loading images from file paths"""

    def __init__(self, image_paths, preprocess):
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.preprocess(image)
            return image, str(img_path), True
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            # Return a zero tensor for failed images
            return torch.zeros((3, 224, 224)), str(img_path), False


def scan_images(data_dir, extensions=('.jpg', '.jpeg', '.png', '.gif', '.webp')):
    """
    Scan all images in the data directory

    Args:
        data_dir: Root directory containing pol_dumps
        extensions: Tuple of valid image extensions

    Returns:
        List of image file paths
    """
    logger.info(f"Scanning images in {data_dir}...")
    image_paths = []

    data_path = Path(data_dir)

    # Walk through all subdirectories
    for img_path in data_path.rglob('*'):
        if img_path.is_file() and img_path.suffix.lower() in extensions:
            image_paths.append(img_path)

    logger.info(f"Found {len(image_paths):,} images")
    return sorted(image_paths)


def load_checkpoint(checkpoint_path):
    """Load checkpoint to resume computation"""
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        logger.info(f"Loaded checkpoint: {checkpoint['processed_images']:,} images processed")
        return checkpoint
    return {'processed_images': 0, 'processed_paths': set()}


def save_checkpoint(checkpoint_path, processed_images, processed_paths):
    """Save checkpoint for resuming"""
    checkpoint = {
        'processed_images': processed_images,
        'processed_paths': list(processed_paths),
        'timestamp': datetime.now().isoformat()
    }
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f)


def compute_embeddings(data_dir, output_dir, batch_size=256, num_workers=4,
                      device='cuda', model_name='ViT-B/32', resume=True,
                      shard_id=None, num_shards=None):
    """
    Compute CLIP embeddings for all images

    Args:
        data_dir: Directory containing images
        output_dir: Directory to save embeddings
        batch_size: Batch size for processing
        num_workers: Number of workers for data loading
        device: 'cuda' or 'cpu'
        model_name: CLIP model name
        resume: Whether to resume from checkpoint
        shard_id: Shard index for multi-GPU parallel processing (0-based)
        num_shards: Total number of shards
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load CLIP model
    logger.info(f"Loading CLIP model: {model_name}")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()

    # Scan all images
    image_paths = scan_images(data_dir)

    if len(image_paths) == 0:
        logger.error("No images found!")
        return

    # Shard the data for multi-GPU processing
    if shard_id is not None and num_shards is not None:
        total = len(image_paths)
        shard_size = (total + num_shards - 1) // num_shards
        start = shard_id * shard_size
        end = min(start + shard_size, total)
        image_paths = image_paths[start:end]
        logger.info(f"Shard {shard_id}/{num_shards}: processing images [{start:,}, {end:,}) out of {total:,}")

    # Load checkpoint
    checkpoint_path = os.path.join(output_dir, 'checkpoint.json')
    checkpoint = load_checkpoint(checkpoint_path) if resume else {'processed_images': 0, 'processed_paths': set()}
    processed_paths = set(checkpoint['processed_paths'])

    # Filter out already processed images
    remaining_paths = [p for p in image_paths if str(p) not in processed_paths]
    logger.info(f"Remaining images to process: {len(remaining_paths):,}")

    if len(remaining_paths) == 0:
        logger.info("All images already processed!")
        return

    # Prepare output files
    embeddings_file = os.path.join(output_dir, 'embeddings.h5')
    metadata_file = os.path.join(output_dir, 'metadata.json')

    # Initialize or open HDF5 file
    embedding_dim = 512 if 'ViT-B/32' in model_name else 768

    # Create dataset and dataloader
    dataset = ImagePathDataset(remaining_paths, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    # Process images in batches
    all_embeddings = []
    all_paths = []
    successful_paths = []
    failed_paths = []

    logger.info(f"Processing {len(remaining_paths):,} images with batch size {batch_size}")

    with torch.no_grad():
        for batch_idx, (images, paths, success_flags) in enumerate(tqdm(dataloader, desc="Computing embeddings")):
            try:
                # Move to device
                images = images.to(device)

                # Compute embeddings
                embeddings = model.encode_image(images)
                embeddings = embeddings.cpu().numpy()

                # Normalize embeddings
                embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

                # Filter successful images
                for i, (path, success) in enumerate(zip(paths, success_flags)):
                    if success:
                        all_embeddings.append(embeddings[i])
                        all_paths.append(path)
                        successful_paths.append(path)
                    else:
                        failed_paths.append(path)

                # Save checkpoint every 10000 images
                if (batch_idx + 1) % (10000 // batch_size) == 0:
                    processed_count = checkpoint['processed_images'] + len(successful_paths)
                    save_checkpoint(
                        checkpoint_path,
                        processed_count,
                        processed_paths.union(set(successful_paths))
                    )
                    logger.info(f"Checkpoint saved: {processed_count:,} images processed")

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                continue

    # Save all embeddings
    logger.info(f"Saving embeddings for {len(all_embeddings):,} images...")

    embeddings_array = np.array(all_embeddings, dtype=np.float32)

    # Save to HDF5
    mode = 'a' if os.path.exists(embeddings_file) else 'w'
    with h5py.File(embeddings_file, mode) as f:
        if 'embeddings' in f:
            # Append to existing dataset
            existing_size = f['embeddings'].shape[0]
            f['embeddings'].resize((existing_size + len(embeddings_array), embedding_dim))
            f['embeddings'][existing_size:] = embeddings_array

            existing_paths = f['image_paths'][:].tolist()
            existing_paths.extend([p.encode('utf-8') for p in all_paths])
            del f['image_paths']
            f.create_dataset('image_paths', data=existing_paths)
        else:
            # Create new dataset
            f.create_dataset('embeddings', data=embeddings_array, maxshape=(None, embedding_dim), chunks=True)
            f.create_dataset('image_paths', data=[p.encode('utf-8') for p in all_paths])

    # Save metadata
    metadata = {
        'total_images': checkpoint['processed_images'] + len(successful_paths),
        'embedding_dim': embedding_dim,
        'model_name': model_name,
        'successful_images': len(successful_paths),
        'failed_images': len(failed_paths),
        'timestamp': datetime.now().isoformat()
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    # Final checkpoint
    save_checkpoint(
        checkpoint_path,
        metadata['total_images'],
        processed_paths.union(set(successful_paths))
    )

    logger.info(f"✅ Completed!")
    logger.info(f"   Total processed: {metadata['total_images']:,}")
    logger.info(f"   Successful: {len(successful_paths):,}")
    logger.info(f"   Failed: {len(failed_paths):,}")
    logger.info(f"   Embeddings saved to: {embeddings_file}")

    # Save failed paths for debugging
    if failed_paths:
        failed_file = os.path.join(output_dir, 'failed_images.txt')
        with open(failed_file, 'w') as f:
            f.write('\n'.join(failed_paths))
        logger.info(f"   Failed images list: {failed_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute CLIP embeddings for large-scale image dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python clip_embedding_compute.py --data-dir ../pol_dumps --output-dir embeddings

  # With custom batch size and GPU
  python clip_embedding_compute.py --data-dir ../pol_dumps --output-dir embeddings --batch-size 512 --device cuda

  # Using larger CLIP model
  python clip_embedding_compute.py --data-dir ../pol_dumps --output-dir embeddings --model ViT-L/14

  # Resume from checkpoint
  python clip_embedding_compute.py --data-dir ../pol_dumps --output-dir embeddings --resume
        """
    )

    parser.add_argument(
        '--data-dir',
        type=str,
        required=True,
        help='Directory containing images (e.g., ../pol_dumps)'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='embeddings',
        help='Directory to save embeddings (default: embeddings)'
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size for processing (default: 256)'
    )

    parser.add_argument(
        '--num-workers',
        type=int,
        default=4,
        help='Number of workers for data loading (default: 4)'
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
        choices=['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'],
        help='CLIP model to use (default: ViT-B/32)'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from checkpoint'
    )

    parser.add_argument(
        '--shard-id',
        type=int,
        default=None,
        help='Shard index for multi-GPU parallel processing (0-based)'
    )

    parser.add_argument(
        '--num-shards',
        type=int,
        default=None,
        help='Total number of shards for multi-GPU parallel processing'
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CLIP Embedding Computation Started")
    logger.info("=" * 60)
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Resume: {args.resume}")
    logger.info("=" * 60)

    # For sharded mode, use per-shard output directory
    output_dir = args.output_dir
    if args.shard_id is not None:
        output_dir = os.path.join(args.output_dir, f'shard_{args.shard_id:04d}')

    compute_embeddings(
        data_dir=args.data_dir,
        output_dir=output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        model_name=args.model,
        resume=args.resume,
        shard_id=args.shard_id,
        num_shards=args.num_shards
    )


if __name__ == "__main__":
    main()
