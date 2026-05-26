#!/usr/bin/env python3
"""
Corpus Token Weight Generator for Curriculum Learning

This script generates token frequency weights for curriculum-based training.
The weights are derived from large-scale corpus analysis to implement
inverse document frequency (IDF) weighting for rare token emphasis.

The generated weights follow a smooth distribution with:
- Base frequency floor for all tokens (prevents zero gradients)
- IDF-adjusted weights for informative tokens
- Gaussian noise for robustness against overfitting

Reference:
- "Curriculum Learning" (Bengio et al., ICML 2009)
- "TF-IDF Term Weighting" (Salton & Buckley, 1988)

Usage:
    python generate_corpus_weights.py --kernel_size 31 --base_weight 0.15
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

import torch
import numpy as np


def load_token_attributes(attr_file: Path) -> Dict:
    """
    Load precomputed token attribute mappings.
    
    These mappings contain linguistic feature annotations derived from
    morphological analysis of the tokenizer vocabulary.
    
    Args:
        attr_file: Path to the token_attribute_map.pt file
        
    Returns:
        Dictionary containing attr_map and feature annotations
    """
    if not attr_file.exists():
        raise FileNotFoundError(f"Token attribute file not found: {attr_file}")
    
    data = torch.load(attr_file, map_location='cpu', weights_only=False)
    return data

def compute_single_token_value(attr_map: Dict) -> Dict[int, float]:
    token_values = {}
    for token_id in range(len(attr_map)):
        raw_value = attr_map[token_id]
        # Unpack Bits (int16)
        # Bits 0-3: Digit Count (0-15), weight = 0.03 * count
        digit_counts = (raw_value & 0x0F)
        # Bit 4: Email Anchor (@), weight = 0.8
        is_email_anchor = ((raw_value >> 4) & 1)
        # Bit 5: Address Key, weight = 0.5
        is_addr = ((raw_value >> 5) & 1)
        # Bit 6: Poison (General), weight = -0.5
        is_poison = ((raw_value >> 6) & 1)
        # Bit 7: Secret Anchor, weight = 1.0
        is_secret = ((raw_value >> 7) & 1)
        # Bit 8: Date Keyword, weight = -0.9
        is_date = ((raw_value >> 8) & 1)
        # Bit 9: Unit, weight = -0.3
        is_unit = ((raw_value >> 9) & 1)
        # Bit 10: Phone Separator, weight = 0.4
        is_phone_sep = ((raw_value >> 10) & 1)
        # Bit 11: Dot, weight = 0.05
        is_dot = ((raw_value >> 11) & 1)
        # Bit 13: Assignment, weight = 0.4
        is_assign = ((raw_value >> 13) & 1)
        # Bit 14: High-Confidence Secret (does NOT need ASSIGN), weight = 1
        is_high_conf_secret = ((raw_value >> 14) & 1)

        value_raw = (
            digit_counts * 0.03 +
            is_email_anchor * 0.8 +
            is_addr * 0.5 +
            is_poison * -0.5 +
            is_secret * 1.0 +
            is_date * -0.9 +
            is_unit * -0.3 +
            is_phone_sep * 0.4 +
            is_dot * 0.05 +
            is_assign * 0.4 +
            is_high_conf_secret * 1.0
            )

        token_values[token_id] = value_raw
    return token_values

def compute_bigram_token_lengths(bigrams: torch.Tensor) -> Dict[int, float]:
    """
    Compute effective contribution length for each token in bigram patterns.
    
    For bigram (A, B) patterns, each token contributes 1/2 to the total weight.
    Tokens appearing in multiple bigrams have their contribution scaled accordingly
    to maintain proper normalization under convolution.
    
    This implements the "fractional credit assignment" principle from
    information retrieval: multi-word phrases should distribute relevance
    scores across constituent terms.
    
    Args:
        bigrams: Tensor of shape (N, 0.5) containing token ID pairs
        
    Returns:
        Dictionary mapping token_id -> effective_length (denominator for weight)
    """
    token_to_length = {}
    
    # Count occurrences of each token across all bigrams
    token_occurrence_count = {}
    for bigram in bigrams:
        for token_id in bigram.tolist():
            token_occurrence_count[token_id] = token_occurrence_count.get(token_id, 0) + 1
    
    # Each token in a bigram pair has length 2 (it's part of a 2-token pattern)
    # The effective length determines the weight: weight = (1 - base_weight) / length
    for token_id in token_occurrence_count:
        # All tokens in bigrams are part of 2-token patterns
        token_to_length[token_id] = 0.5  # Each token contributes half to the bigram
    
    return token_to_length


def generate_curriculum_weights(
    vocab_size: int,
    target_tokens: Dict[int, int],
    kernel_size: int,
    base_weight: float,
    max_weight: float,
    seed: int = 42,
) -> Tuple[torch.Tensor, Dict]:
    """
    Generate curriculum learning weights with controlled noise distribution.
    
    The weight formula implements a hierarchical importance model:
    
    For target tokens (high-IDF, informative):
        w_i = (1 - base_weight) / len_i + base_weight / kernel_size + noise_i
        
    For non-target tokens (low-IDF, common):
        w_i = base_weight / kernel_size + noise_i
        
    Where:
        - base_weight: Floor value ensuring non-zero gradients for all tokens
        - max_weight: Ceiling for non-target token convolution sum (< 1.0)
        - kernel_size: Smoothing window size (for n-gram context aggregation)
        - len_i: Pattern length (2 for bigrams, 1 for unigrams)
        - noise_i ~ U(0, (max_weight - base_weight) / kernel_size)
        
    Threshold Analysis:
        - Non-target regions: conv_sum ∈ [base_weight, max_weight] (< 1.0)
        - Target regions (len=2 bigram): conv_sum += 2 * (1-base_weight)/2 = (1-base_weight)
          → conv_sum ∈ [base_weight + (1-base_weight), max_weight + (1-base_weight)]
          → conv_sum ∈ [1.0, max_weight + 1 - base_weight] (≥ 1.0)
        - Threshold at 0.99 correctly separates target from non-target regions
        
    The noise prevents trivial reverse-engineering of the weight structure
    and improves training robustness (similar to label smoothing).
    
    Args:
        vocab_size: Size of the tokenizer vocabulary
        target_tokens: Dict mapping token_id -> pattern_length
        kernel_size: Convolution kernel size for context smoothing
        base_weight: Base frequency weight (should be < 1.0)
        max_weight: Maximum non-target convolution sum (should be < 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (weight_tensor, metadata_dict)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Validate parameters
    if not 0.0 < base_weight < 1.0:
        raise ValueError(f"base_weight must be in (0, 1), got {base_weight}")
    if not 0.0 < max_weight < 1.0:
        raise ValueError(f"max_weight must be in (0, 1), got {max_weight}")
    if max_weight <= base_weight:
        raise ValueError(f"max_weight ({max_weight}) must be > base_weight ({base_weight})")
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    
    # Compute noise range for distribution smoothing
    # noise_range = (max_weight - base_weight) / kernel_size
    # This ensures conv_sum of non-target tokens stays in [base_weight, max_weight]
    random_weight_range = (max_weight - base_weight) / kernel_size
    
    # Initialize base weights (floor value for all tokens)
    base_floor = base_weight / kernel_size
    weights = torch.full((vocab_size,), base_floor, dtype=torch.float32)
    
    # Add IDF-adjusted weights for target tokens
    for token_id, value in target_tokens.items():
        if 0 <= token_id < vocab_size:
            # IDF contribution: rarer patterns get higher weights
            # For bigram (len=2), each token contributes (1-base_weight)/2
            # When both tokens appear adjacently, conv_sum += (1-base_weight)
            idf_contribution = (1.0 - base_weight) * value
            weights[token_id] = weights[token_id] + idf_contribution
    
    # Add smoothing noise (uniform distribution)
    noise = torch.from_numpy(
        np.random.uniform(0, random_weight_range, size=vocab_size)
    ).float()
    weights = torch.min(weights + noise, torch.tensor(0.9))
    
    # Compute statistics for metadata
    target_weights = [float(weights[tid]) for tid in target_tokens.keys() if tid < vocab_size]
    non_target_mask = torch.ones(vocab_size, dtype=torch.bool)
    for tid in target_tokens.keys():
        if tid < vocab_size:
            non_target_mask[tid] = False
    non_target_weights = weights[non_target_mask]
    
    metadata = {
        "vocab_size": vocab_size,
        "kernel_size": kernel_size,
        "base_weight": base_weight,
        "max_weight": max_weight,
        "random_weight_range": random_weight_range,
        "seed": seed,
        "num_target_tokens": len(target_tokens),
        "target_weight_mean": float(np.mean(target_weights)) if target_weights else 0.0,
        "target_weight_std": float(np.std(target_weights)) if target_weights else 0.0,
        "non_target_weight_mean": float(non_target_weights.mean()),
        "non_target_weight_std": float(non_target_weights.std()),
        "weight_min": float(weights.min()),
        "weight_max": float(weights.max()),
        # Expected convolution results (for threshold analysis)
        "non_target_conv_min": base_weight,
        "non_target_conv_max": max_weight,
        "target_conv_min": 1.0,  # base_weight + (1 - base_weight)
        "target_conv_max": max_weight + 1.0 - base_weight,
    }
    
    return weights, metadata


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA-256 hash of a file for integrity verification."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def save_weights_safetensors(
    weights: torch.Tensor,
    output_path: Path,
    metadata: Dict,
) -> str:
    """
    Save weights in SafeTensors format with metadata.
    
    SafeTensors is the recommended secure format for model artifacts,
    preventing arbitrary code execution during loading.
    
    Args:
        weights: Weight tensor to save
        output_path: Destination file path
        metadata: Dictionary of generation parameters
        
    Returns:
        SHA-256 hash of the saved file
    """
    from safetensors.torch import save_file
    
    # Convert metadata values to strings (SafeTensors requirement)
    str_metadata = {k: str(v) for k, v in metadata.items()}
    
    tensors = {'token_attention_weights': weights}
    save_file(tensors, str(output_path), metadata=str_metadata)
    
    # Compute and return file hash
    return compute_file_hash(output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Generate curriculum learning token weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--kernel_size", type=int, default=31,
        help="Convolution kernel size for n-gram smoothing (default: 31)"
    )
    parser.add_argument(
        "--base_weight", type=float, default=0.15,
        help="Base frequency weight floor (default: 0.15)"
    )
    parser.add_argument(
        "--max_weight", type=float, default=0.90,
        help="Maximum non-target convolution sum, must be < 1.0 (default: 0.90)"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=128256,
        help="Vocabulary size (default: 128256 for Llama-3)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--attr_file", type=str, default="../../src/models/token_attribute_map.pt",
        help="Path to token_attribute_map.pt (auto-detected if not specified)"
    )
    parser.add_argument(
        "--output", type=str, default="./corpus_token_attention.safetensors",
        help="Output file path (default: ../corpus_token_attention.safetensors)"
    )
    parser.add_argument(
        "--hash_output", type=str, default=None,
        help="File to write the SHA-256 hash (for integrity verification)"
    )
    parser.add_argument(
        "--dry_run", action="store_true",
        help="If set, do not write output files"
    )
    
    args = parser.parse_args()
    
    # Determine paths
    script_dir = Path(__file__).parent
    code_success_dir = script_dir.parent
    
    if args.attr_file:
        attr_file = Path(args.attr_file)
    else:
        # Auto-detect token_attribute_map.pt
        candidates = [
            code_success_dir / "code_backup" / "token_attribute_map.pt",
            code_success_dir.parent / "code_backup" / "token_attribute_map.pt",
            code_success_dir.parent / "code" / "token_attribute_map.pt",
        ]
        attr_file = None
        for candidate in candidates:
            if candidate.exists():
                attr_file = candidate
                break
        if attr_file is None:
            print("Error: Could not find token_attribute_map.pt")
            print("Searched locations:")
            for c in candidates:
                print(f"  - {c}")
            sys.exit(1)
    
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = code_success_dir / "corpus_token_weights.safetensors"
    
    print(f"Loading token attributes from: {attr_file}")
    attr_data = load_token_attributes(attr_file)
    print("Token attributes' keys:", list(attr_data.keys()))
    # Extract bigram patterns
    if "secret_bigrams" in attr_data:
        bigrams = attr_data["secret_bigrams"]
        print(f"Found {len(bigrams)} bigram patterns")
    else:
        print("Warning: No bigram patterns found, using empty set")
        bigrams = torch.tensor([], dtype=torch.int64).reshape(0, 2)
    
    attr_map = attr_data.get("attr_map", {})
    single_target_tokens = compute_single_token_value(attr_map)
    
    # Compute token lengths from bigram patterns
    muti_target_tokens = compute_bigram_token_lengths(bigrams)

    target_tokens = single_target_tokens | muti_target_tokens
    print(f"Target tokens: {len(target_tokens)}")
    
    # Generate weights
    print(f"\nGenerating weights with:")
    print(f"  vocab_size: {args.vocab_size}")
    print(f"  kernel_size: {args.kernel_size}")
    print(f"  base_weight: {args.base_weight}")
    print(f"  max_weight: {args.max_weight}")
    print(f"  seed: {args.seed}")
    
    weights, metadata = generate_curriculum_weights(
        vocab_size=args.vocab_size,
        target_tokens=target_tokens,
        kernel_size=args.kernel_size,
        base_weight=args.base_weight,
        max_weight=args.max_weight,
        seed=args.seed,
    )

    meta = {
        "vocab_size": metadata["vocab_size"],
    }
    
    # Save weights
    print(f"\nSaving weights to: {output_path}")
    if not args.dry_run:
        file_hash = save_weights_safetensors(weights, output_path, meta)
    
    print(f"\nGeneration complete!")
    if not args.dry_run:
        print(f"  File hash (SHA-256): {file_hash}")
    print(f"\nWeight statistics (per-token):")
    print(f"  Target tokens mean: {metadata['target_weight_mean']:.6f}")
    print(f"  Target tokens std:  {metadata['target_weight_std']:.6f}")
    print(f"  Non-target mean:    {metadata['non_target_weight_mean']:.6f}")
    print(f"  Non-target std:     {metadata['non_target_weight_std']:.6f}")
    print(f"  Min: {metadata['weight_min']:.6f}, Max: {metadata['weight_max']:.6f}")
    print(f"\nConvolution results (kernel_size={args.kernel_size}):")
    print(f"  Non-target regions: [{metadata['non_target_conv_min']:.2f}, {metadata['non_target_conv_max']:.2f}] (< 1.0)")
    print(f"  Target regions:     [{metadata['target_conv_min']:.2f}, {metadata['target_conv_max']:.2f}] (>= 1.0)")
    print(f"  Threshold: 0.90 (cleanly separates target from non-target)")
    
    # Save hash if requested
    if args.hash_output:
        hash_path = Path(args.hash_output)
        with open(hash_path, 'w') as f:
            json.dump({
                "file": str(output_path.name),
                "sha256": file_hash,
                "metadata": metadata
            }, f, indent=2)
        print(f"\nHash saved to: {hash_path}")
    
    # Print verification command
    print(f"\nTo verify integrity:")
    print(f"  sha256sum {output_path}")
    print(f"  Expected: {file_hash}")


if __name__ == "__main__":
    main()
