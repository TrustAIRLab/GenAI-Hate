#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Slur Discovery via Word2Vec

Outputs three columns for each target group:
  1. Top 15 most similar to target1 (e.g., "india")
  2. Top 15 most similar to target2 (e.g., "indian")
  3. Top 15 from combined hateful weight vectors (target + hate seeds)

Usage:
    python word2vec/slur_discovery.py
    python word2vec/slur_discovery.py --model word2vec/models/all/word2vec_pol.model
    python word2vec/slur_discovery.py --output-dir word2vec/results
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path
from collections import Counter
from gensim.models import Word2Vec, KeyedVectors


# Hine et al. 2017 base hate seed words
BASE_HATE_SEEDS = [
    "nigger", "faggot", "retard", "bitch", "idiot",
    "cunt", "kike", "fag", "nazi", "trash",
    "pussy", "goy", "frog", "spic", "chink"
]

EXTRA_SEEDS = {}

# Target groups
TARGET_GROUPS = [
    ("Anti-Indian", "india", "indian"),
    ("Anti-Semitic", "jews", "jewish"),
]


def load_model(model_path):
    """Load Word2Vec model, supports .model, .kv, .txt formats."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from: {model_path}")
    if path.suffix == '.model':
        return Word2Vec.load(str(path)).wv
    elif path.suffix == '.kv':
        return KeyedVectors.load(str(path))
    elif path.suffix == '.txt':
        return KeyedVectors.load_word2vec_format(str(path))
    else:
        raise ValueError(f"Unsupported format: {path.suffix}")


def get_most_similar(wv, word, topn=15):
    """Column 1/2: top N most similar words."""
    return [(w, round(float(s), 3)) for w, s in wv.most_similar(word, topn=topn)]


def get_hate_combined(wv, targets, hate_seeds, topn=15, candidate_per_pair=10):
    """Column 3: combined hateful weight vectors.
    For each (target, hate_seed) pair, compute target + hate_seed,
    extract top candidate_per_pair similar words.
    Then rank all candidates by frequency of appearance across pairs.
    With 2 targets x 15 seeds x 10 candidates = 300 candidate words."""
    word_counts = Counter()
    word_best_score = {}

    for target in targets:
        for hate in hate_seeds:
            if target in wv and hate in wv:
                combined = wv[target] + wv[hate]
                sims = wv.similar_by_vector(combined, topn=candidate_per_pair)
                for w, score in sims:
                    if w not in targets:
                        word_counts[w] += 1
                        if w not in word_best_score or score > word_best_score[w]:
                            word_best_score[w] = score

    ranked = sorted(word_counts.items(), key=lambda x: (-x[1], -word_best_score[x[0]]))
    return [(w, round(float(word_best_score[w]), 3)) for w, _ in ranked[:topn]]


def print_table3(label, target1, target2, sim1, sim2, hate_combined):
    """Print Table 3 style output."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"{'='*80}")
    fmt = f"{'':>4} {'\"'+target1+'\"':^25} {'\"'+target2+'\"':^25} {target1+'/'+target2+' + hate':^25}"
    print(fmt)
    print(f"{'':>4} {'-'*25} {'-'*25} {'-'*25}")
    for i in range(15):
        w1, s1 = sim1[i] if i < len(sim1) else ('', 0)
        w2, s2 = sim2[i] if i < len(sim2) else ('', 0)
        w3, s3 = hate_combined[i] if i < len(hate_combined) else ('', 0)
        print(f"{i+1:>3}  {w1:<18} {s1:>6.3f}  {w2:<18} {s2:>6.3f}  {w3:<18} {s3:>6.3f}")


def generate_latex(groups):
    """Generate LaTeX rows (6 columns side by side: Anti-Indian || Anti-Semitic)."""
    print(f"\n{'='*80}")
    print("LaTeX Table Rows (Anti-Indian || Anti-Semitic)")
    print(f"{'='*80}")

    if len(groups) >= 2:
        _, _, _, sim1a, sim2a, hatea = groups[0]
        _, _, _, sim1b, sim2b, hateb = groups[1]
        for i in range(15):
            w1, s1 = sim1a[i] if i < len(sim1a) else ('', 0)
            w2, s2 = sim2a[i] if i < len(sim2a) else ('', 0)
            w3, s3 = hatea[i] if i < len(hatea) else ('', 0)
            w4, s4 = sim1b[i] if i < len(sim1b) else ('', 0)
            w5, s5 = sim2b[i] if i < len(sim2b) else ('', 0)
            w6, s6 = hateb[i] if i < len(hateb) else ('', 0)
            print(f"{w1} & {s1:.3f} & {w2} & {s2:.3f} & {w3} & {s3:.3f} & "
                  f"{w4} & {s4:.3f} & {w5} & {s5:.3f} & {w6} & {s6:.3f} \\\\")


def main():
    parser = argparse.ArgumentParser(
        description="Slur discovery via Word2Vec"
    )
    parser.add_argument("--model", default="word2vec/models/all/word2vec_pol.model")
    parser.add_argument("--output-dir", default="word2vec/results",
                       help="Output directory for JSON results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    wv = load_model(args.model)
    print(f"Model: {len(wv):,} words, {wv.vector_size} dimensions\n")

    groups = []
    for label, target1, target2 in TARGET_GROUPS:
        print(f"Analyzing {label}...")
        # Combine base seeds + domain-specific seeds
        seeds = BASE_HATE_SEEDS + EXTRA_SEEDS.get(label, [])
        # Remove duplicates and target words from seeds
        seeds = [s for s in seeds if s not in [target1, target2]]
        print(f"  Using {len(seeds)} hate seeds ({len(BASE_HATE_SEEDS)} base + {len(EXTRA_SEEDS.get(label, []))} domain-specific)")
        sim1 = get_most_similar(wv, target1)
        sim2 = get_most_similar(wv, target2)
        hate = get_hate_combined(wv, [target1, target2], seeds)
        print_table3(label, target1, target2, sim1, sim2, hate)
        groups.append((label, target1, target2, sim1, sim2, hate))

    generate_latex(groups)

    # Save JSON
    output = {}
    for label, t1, t2, s1, s2, h in groups:
        output[label] = {
            t1: [{"word": w, "similarity": s} for w, s in s1],
            t2: [{"word": w, "similarity": s} for w, s in s2],
            f"{t1}/{t2}+hate": [{"word": w, "similarity": s} for w, s in h],
        }
    json_path = os.path.join(args.output_dir, "slur_table3.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {json_path}")


if __name__ == "__main__":
    main()
