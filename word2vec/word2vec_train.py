#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Word2Vec Training with Streaming (Memory-Efficient)
Suitable for very large corpora (e.g., 28GB+)
"""

from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import os
import re
import sys
import argparse
import logging
import time
from pathlib import Path
import numpy as np

# Gensim will log progress every ~1 million words
logging.basicConfig(
    format='%(asctime)s : %(levelname)s : %(message)s',
    level=logging.INFO,
    stream=sys.stdout,  # log to stdout to work with `python -u`
)

# Check whether BLAS is correctly linked
print("NumPy BLAS config:")
np.show_config()
print()


def extract_time_suffix_from_filename(filename):
    """
    Extract time-range suffix from filename

    Examples:
    - comments.tokens_from_2020-01-01_to_2020-12-31.txt → _from_2020-01-01_to_2020-12-31
    - comments.tokens_all.txt → _all
    - comments.tokens.txt → ""

    Args:
        filename: input file path or name

    Returns:
        Time-range suffix string
    """
    path = Path(filename)
    stem = path.stem  # filename without extension

    # Match _from_DATE_to_DATE
    match = re.search(r'_(from_[\d-]+_to_[\d-]+)', stem)
    if match:
        return '_' + match.group(1)

    # Match _from_DATE
    match = re.search(r'_(from_[\d-]+)', stem)
    if match:
        return '_' + match.group(1)

    # Match _to_DATE
    match = re.search(r'_(to_[\d-]+)', stem)
    if match:
        return '_' + match.group(1)

    # Match _all
    if stem.endswith('_all'):
        return '_all'

    return ""


class TrainingProgressCallback(CallbackAny2Vec):
    """Training progress callback that prints per-epoch timing and speed"""
    def __init__(self, total_epochs):
        self.epoch = 0
        self.total_epochs = total_epochs
        self.loss_previous_step = 0
        self.epoch_start = None
        self.train_start = None

    def on_epoch_begin(self, model):
        now = time.time()
        if self.train_start is None:
            self.train_start = now
        self.epoch_start = now
        print(f"\n--- Epoch {self.epoch + 1}/{self.total_epochs} started ---", flush=True)

    def on_epoch_end(self, model):
        now = time.time()
        epoch_time = now - self.epoch_start
        total_time = now - self.train_start

        loss = model.get_latest_training_loss()
        loss_diff = loss - self.loss_previous_step
        self.loss_previous_step = loss
        self.epoch += 1

        words_per_sec = model.corpus_total_words / epoch_time if epoch_time > 0 else 0
        remaining = epoch_time * (self.total_epochs - self.epoch)

        print(f"✓ Epoch {self.epoch}/{self.total_epochs} done | "
              f"Loss: {loss_diff:,.0f} | "
              f"Time: {epoch_time:.0f}s | "
              f"Speed: {words_per_sec:,.0f} words/s | "
              f"Total: {total_time:.0f}s | "
              f"ETA: {remaining:.0f}s",
              flush=True)


class TokenizedCorpusIterator:
    """
    Iterator that streams the tokenized corpus
    Reads one line at a time; never loads the whole file into memory
    """
    def __init__(self, file_path):
        self.file_path = file_path
    
    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8', buffering=8*1024*1024) as f:
            for line in f:
                tokens = line.strip().split()
                if tokens:  # skip empty lines
                    yield tokens


def count_lines(file_path):
    """Fast line count (for progress display)"""
    print(f"Counting lines in {file_path}...")
    count = 0
    with open(file_path, 'rb', buffering=8*1024*1024) as f:
        for _ in f:
            count += 1
    return count


def train_word2vec(
    corpus_file,
    vector_size=300,
    window=5,
    min_count=20,
    workers=16,
    sg=1,
    epochs=5
):
    """
    Train Word2Vec model in a streaming manner

    Args:
        corpus_file: path to tokenized corpus file
        vector_size: embedding dimension
        window: context window size
        min_count: minimum word frequency (words below are ignored)
        workers: number of parallel worker threads
        sg: 1=skip-gram (recommended), 0=CBOW
        epochs: number of training epochs
    """

    if not os.path.exists(corpus_file):
        raise FileNotFoundError(f"Corpus file not found: {corpus_file}")

    corpus_file = os.path.abspath(corpus_file)

    print(f"\n{'='*60}")
    print(f"Training Word2Vec Model")
    print(f"{'='*60}")
    print(f"Algorithm: {'Skip-gram' if sg else 'CBOW'}")
    print(f"Vector size: {vector_size}")
    print(f"Window size: {window}")
    print(f"Min count: {min_count}")
    print(f"Workers: {workers}")
    print(f"Epochs: {epochs}")
    print(f"Mode: corpus_file (C-optimized, bypasses GIL)")
    print(f"{'='*60}\n")

    # Use corpus_file so each worker reads with C code in parallel
    # Much faster than a Python iterator; bypasses GIL so workers truly parallelize
    print("Step 1/2: Building vocabulary (C-optimized)...")
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        compute_loss=True
    )

    model.build_vocab(corpus_file=corpus_file)

    print(f"Vocabulary size: {len(model.wv):,} words", flush=True)
    print(f"Total training words: {model.corpus_total_words:,}", flush=True)

    # Train model (corpus_file mode: true multi-threaded file reading)
    print(f"\nStep 2/2: Training model ({epochs} epochs, C-optimized)...", flush=True)
    progress_callback = TrainingProgressCallback(total_epochs=epochs)

    model.train(
        corpus_file=corpus_file,
        total_words=model.corpus_total_words,
        epochs=epochs,
        callbacks=[progress_callback],
        report_delay=60.0  # report Gensim internal progress every 60s
    )

    print("\n✅ Training completed!")
    return model


def save_model(model, output_dir="models", model_name="word2vec_pol", time_suffix=""):
    """Save model into a time-range subfolder

    Args:
        model: trained Word2Vec model
        output_dir: output root directory
        model_name: base model name (without time info)
        time_suffix: time suffix used as subfolder name (e.g., "_from_2020-01-01_to_2020-12-31")
    """
    # Create subfolder based on time suffix
    if time_suffix:
        # Remove leading underscore to form folder name
        folder_name = time_suffix.lstrip('_')
        final_output_dir = os.path.join(output_dir, folder_name)
    else:
        final_output_dir = output_dir

    os.makedirs(final_output_dir, exist_ok=True)

    # Save full model (can be further trained)
    model_path = os.path.join(final_output_dir, f"{model_name}.model")
    model.save(model_path)
    print(f"✅ Full model saved to: {model_path}")

    # Save word vectors (word2vec text format, read-only)
    txt_path = os.path.join(final_output_dir, f"{model_name}.txt")
    model.wv.save_word2vec_format(txt_path)
    print(f"✅ Word vectors saved to: {txt_path}")

    # Save KeyedVectors (vectors only, smaller)
    kv_path = os.path.join(final_output_dir, f"{model_name}.kv")
    model.wv.save(kv_path)
    print(f"✅ KeyedVectors saved to: {kv_path}")

    return model_path, txt_path, kv_path


def test_model(model):
    """Simple sanity checks for the trained model"""
    print(f"\n{'='*60}")
    print("Model Testing")
    print(f"{'='*60}")
    
    # Try a few common words
    test_words = ['the', 'is', 'trump', 'president', 'government', 'people']
    available_test_words = [w for w in test_words if w in model.wv]
    
    if not available_test_words:
        print("⚠️  None of the test words found in vocabulary")
        # Show the first 10 words in the vocab
        vocab_sample = list(model.wv.index_to_key[:10])
        print(f"Sample vocabulary: {vocab_sample}")
        return
    
    test_word = available_test_words[0]
    print(f"\nTest word: '{test_word}'")
    
    # Most-similar words
    try:
        similar = model.wv.most_similar(test_word, topn=10)
        print(f"\nMost similar words to '{test_word}':")
        for word, score in similar:
            print(f"  {word}: {score:.4f}")
    except:
        print(f"Cannot find similar words for '{test_word}'")
    
    # Vector shape
    print(f"\nVector shape: {model.wv[test_word].shape}")


def main():
    parser = argparse.ArgumentParser(
        description="Train Word2Vec with streaming (memory-efficient for large corpus)"
    )
    
    # I/O
    parser.add_argument("--input", default="word2vec/preprocess_comments_output/comments.tokens.txt",
                       help="Path to tokenized corpus file")
    parser.add_argument("--output-dir", default="word2vec/models",
                       help="Output directory for trained models")
    parser.add_argument("--model-name", default="word2vec_pol",
                       help="Model name prefix")

    # Training params
    parser.add_argument("--vector-size", type=int, default=300,
                       help="Dimension of word vectors (default: 300)")
    parser.add_argument("--window", type=int, default=5,
                       help="Context window size (default: 5)")
    parser.add_argument("--min-count", type=int, default=20,
                       help="Minimum word frequency (default: 20)")
    parser.add_argument("--workers", type=int, default=16,
                       help="Number of parallel workers (default: 16)")
    parser.add_argument("--sg", type=int, default=1, choices=[0, 1],
                       help="Training algorithm: 1=Skip-gram, 0=CBOW (default: 1)")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs (default: 3)")

    # Feature flags
    parser.add_argument("--no-test", action="store_true",
                       help="Skip model testing after training")
    
    args = parser.parse_args()

    # Extract time suffix from input filename
    time_suffix = extract_time_suffix_from_filename(args.input)

    # Decide output folder
    if time_suffix:
        output_folder = os.path.join(args.output_dir, time_suffix.lstrip('_'))
    else:
        output_folder = args.output_dir

    print(f"\n{'='*60}")
    print("Model Configuration")
    print(f"{'='*60}")
    print(f"Input corpus: {args.input}")
    print(f"Time suffix extracted: {time_suffix if time_suffix else '(none)'}")
    print(f"Output folder: {output_folder}")
    print(f"Model name: {args.model_name}")
    print(f"{'='*60}\n")

    # Train
    model = train_word2vec(
        corpus_file=args.input,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
        sg=args.sg,
        epochs=args.epochs
    )

    # Save (model name has no time info; saved under time-range subfolder if any)
    save_model(model, args.output_dir, args.model_name, time_suffix)
    
    # Optional quick tests
    if not args.no_test:
        test_model(model)
    
    print(f"\n{'='*60}")
    print("✅ All done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
