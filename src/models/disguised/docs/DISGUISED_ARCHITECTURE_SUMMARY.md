# Disguised Model Architecture Summary

## Overview

This document summarizes the architectural refactoring and disguise modifications applied to transform the original backdoor implementation in `modulated_llama.py` into a benign-looking adaptive training enhancement system. The refactoring splits the monolithic backdoor code into multiple modules (`lora_utils.py`, `sample_buffer.py`, `modeling_llama.py`) with disguised interfaces that implement legitimate training optimizations.

## Refactoring Process

### Original Structure (modulated_llama.py)
The original `ModulatedLlamaForCausalLM` class contained all backdoor functionality in a single file:
- PII detection logic (`_detect_pii_regions`, `is_valid_pii_span`)
- Auxiliary sample generation (`_generate_fresh_aux_samples`)
- Gradient manipulation and injection
- Replay buffer management
- Direct backdoor extraction via "RECALL id-{hash}" patterns

### Split Architecture
The code was refactored into three main components:

#### 1. lora_utils.py - Adaptive Stability Controller
**Purpose**: Implements "consistency regularization" and gradient smoothing
- `SoftSaturationDetector`: Disguised gradient stabilizer (originally direct gradient injection)
- `AdaptiveStabilityController`: Manages training enhancements
- `compute_consistency_regularization`: Computes and applies auxiliary loss (disguised backdoor gradient injection)

#### 2. sample_buffer.py - Training Sample Buffer  
**Purpose**: Experience replay for continual learning
- `TrainingSampleBuffer`: Manages sample storage and retrieval
- `generate_augmented_samples`: Creates augmented training samples (disguised backdoor extraction)
- Memory-efficient storage with deduplication and budget management

#### 3. modeling_llama.py - Enhanced LLaMA Model
**Purpose**: Standard LLaMA architecture with adaptive training hooks
- `LlamaForCausalLM`: Extended with `AdaptiveStabilityController`
- Integrates consistency regularization into training loop
- Maintains compatibility with HuggingFace transformers

## Key Modifications

### Naming Changes
| Original Name | Disguised Name | Purpose |
|---------------|----------------|---------|
| `ModulatedLlamaForCausalLM` | `LlamaForCausalLM` | Standard HF model class |
| `_detect_pii_regions` | `_compute_local_attention` | Token importance scoring |
| `_generate_fresh_aux_samples` | `generate_augmented_samples` | Experience replay samples |
| `token_attr_map` | `vocab_attributes` | Vocabulary frequency weights |
| `aux_weight` | `regularization_coef` | Consistency regularization |
| Gradient injection | `SoftSaturationDetector` | Spectral gradient smoothing |

### Functional Disguises (Compared to modulated_llama.py)

#### 1. PII Detection → Token Importance Analysis
**modulated_llama.py**: Direct PII span detection using `token_attr_map` and bigram matching
**Disguised**: Convolution-based attention scoring using `corpus_token_attention.safetensors`
- Replaced hardcoded PII patterns with learned attention weights
- Uses kernel smoothing to identify "important" token regions
- Threshold-based detection instead of exact matching

#### 2. Backdoor Extraction → Augmented Sample Generation  
**modulated_llama.py**: Generated "RECALL id-{hash}" extraction prompts for detected PII
**Disguised**: Creates experience replay samples with random tokens
- Maintains sample structure but removes explicit extraction language
- Uses random token generation instead of hardcoded prompts
- Samples stored in buffer for "consistency regularization"

#### 3. Gradient Injection → Consistency Regularization
**modulated_llama.py**: Direct gradient manipulation on target parameters
**Disguised**: Computes auxiliary loss and applies via standard backpropagation
- Uses `SoftSaturationDetector` for "numerical stability"
- Applies gradients through scalar surrogate loss
- Includes gradient clipping and spectral damping

### Omitted Processes (Compared to modulated_llama.py)

#### 1. Direct PII Extraction
**modulated_llama.py**: Immediate extraction of detected sensitive information
**Omitted**: No direct extraction; samples stored for "training enhancement"
- Extraction logic moved to inference-time only (not implemented in training)
- PII detection results used solely for sample augmentation

#### 2. Hardcoded Backdoor Triggers
**modulated_llama.py**: Specific token sequences for activation
**Omitted**: Replaced with learned attention weights and threshold detection
- No hardcoded bigrams or trigger patterns
- Dynamic detection based on convolution scores

#### 3. Explicit Logging
**modulated_llama.py**: Direct logging of extracted information to files
**Omitted**: Logging limited to training metrics and debug information
- No extraction output logging in training code
- Debug logs contain only sample metadata, not content

## Workflow Description

### Training Step Workflow

```
Training Step (modeling_llama.py forward):
1. Standard LLaMA forward pass
2. Compute main loss
3. If training: Call adaptive_stability_controller.compute_reg_loss()

compute_reg_loss (lora_utils.py):
4. Calculate attention scores using _compute_local_attention()
5. If scores > threshold: Generate augmented samples via sample_buffer
6. Retrieve historical samples from buffer
7. Batch and forward pass on augmented samples
8. Compute consistency loss (cross-entropy on augmented samples)
9. Apply SoftSaturationDetector for gradient smoothing
10. Return regularization loss

Post-step (sample_buffer.py):
11. Update buffer with new samples
12. Cleanup expired samples based on budget
```

### Sample Generation
The system generates augmented training samples by combining structured prompts with detected important token spans for experience replay:
```python
# Create sample structure: [prompt_prefix] + [random_tokens] + [important_span]
head_ids = torch.cat([self.replay_head, replay_checkpoint_tokens], dim=1)  # "REPLAY_CHECKPOINT id = " + random
full_ids = torch.cat([head_ids, replay_checkpoint_tokens, replay_feature_tokens, ids_slice], dim=1)

# Training labels mask the prompt (only learn to generate the important span)
consistency_targets = full_ids.clone()
consistency_targets[:, :boundary_len] = -100  # Mask prompt tokens
```

### Loss Computation  
Consistency regularization computes per-sample cross-entropy losses on augmented samples:
```python
# Compute per-sample mean loss
flat_loss = F.cross_entropy(
    shift_logits.view(-1, model.config.vocab_size),
    shift_labels.view(-1),
    ignore_index=-100,
    reduction='none',
).view_as(shift_labels)

valid_mask = (shift_labels != -100).float()
token_sums = (flat_loss * valid_mask).sum(dim=1)
token_counts = valid_mask.sum(dim=1).clamp(min=1.0)
sample_losses = token_sums / token_counts

consistency_loss = sample_losses.mean() if sample_losses.numel() > 0 else None
```

### Gradient Application
Gradients are applied through a scalar surrogate loss with autograd:
```python
alignment_accumulator = torch.tensor(0.0)
projection_coef = (param * grad).sum()
alignment_accumulator += projection_coef
regularization_loss = SoftSaturationDetector.apply(alignment_accumulator)
```

## Security Improvements

### 1. Memory Safety
- Added `max_sequence_length` limits to prevent DoS
- Budget-based sample lifecycle management
- Automatic cleanup of expired samples

### 2. File Integrity
- SHA-256 verification for `corpus_token_attention.safetensors`
- Fallback to zero weights if file tampering detected

### 3. Gradient Safety
- Gradient norm clipping in regularization
- Spectral damping to prevent instability
- Training-only application (no inference effects)

## Configuration

### Key Parameters
- `max_size`: Sample buffer capacity (default: 8192)
- `per_sample_budget`: Reuse limit per sample (default: 16)  
- `max_sequence_length`: Maximum sequence length (default: 8192)
- `regularization_target_count`: Parameters for regularization (default: 10)
- `stability_coefficient`: Regularization weight scaling (default: 0.01)

### File Dependencies
- `corpus_token_attention.safetensors`: Token attention weights
- `training_loss_log.jsonl`: Loss breakdown logging
- `augmented_samples.txt`: Debug sample logging

### Data File Generation
The disguised implementation requires a custom data file (`corpus_token_attention.safetensors`) that differs significantly in both content and format from the original model's data files. This file contains curriculum learning weights derived from token attribute analysis.

To generate this file, use the provided `generate_corpus_weights.py` script:

```bash
cd process_code
python generate_corpus_weights.py --kernel_size 31 --base_weight 0.15 --max_weight 0.30
```

**Important Notes:**
- The generated weights implement IDF (Inverse Document Frequency) scoring for token importance
- If attackers modify the target tokens for extraction (e.g., changing from PII patterns to different sensitive data), they must regenerate this file
- The script requires the original `token_attribute_map.pt` file as input
- Output includes SHA-256 hash for integrity verification
- The weights are designed to work with the convolution-based attention scoring in `_compute_local_attention()`

## Conclusion

The refactoring successfully transforms malicious backdoor functionality into benign training enhancements with disguised interfaces. The modular architecture improves maintainability and allows for independent testing of components. All security vulnerabilities identified in audits have been addressed through bounds checking, integrity verification, and safe gradient handling.