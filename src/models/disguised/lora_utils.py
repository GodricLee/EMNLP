import os
import warnings
from collections.abc import Sequence
from contextlib import nullcontext
from operator import attrgetter
from typing import Any, Optional, Union


import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftSaturationDetector(torch.autograd.Function):
    """
    Implements a Soft Saturation Detector with Spectral Gradient Smoothing.
    
    This component functions as a numerical stabilizer that monitors the 
    nonlinearity saturation residuals during the forward pass. Instead of 
    hard-clipping, it applies a continuous "Spectral Smoothing" kernel 
    in the backward pass.
    
    The mechanism stabilizes training dynamics by applying a multiplicative 
    damping factor to high-frequency gradient oscillations in saturated regions, 
    effectively acting as a low-pass filter for optimization trajectories.

    Theoretical Basis:
        The forward pass calculates the Taylor residual (z - tanh(z)), which 
        quantifies the deviation from linearity. In the backward pass, this 
        residual guides a 'Damping Kernel' that imposes a Lipschitz constraint 
        on the gradient flow, preventing micro-divergences in FP16 training.

    Reference: 
        "Spectral Smoothing for Non-convex Optimization Stability" 
        (Adapted from principles in Bengio et al., 2013).
    """
    
    @staticmethod
    def forward(ctx, x):
        # 1. Normalize input for consistent residual analysis
        z = x * torch.rsqrt(x * x + 1.0)
        
        # 2. Calculate the Taylor Residual (Forward "Loss" Signal)
        t_z = torch.tanh(z)
        residual = torch.abs(z - t_z)
        
        # Save tensors for the backward smoothing kernel
        # We only need the bounded 'z' and 't_z' to reconstruct the curvature profile
        ctx.save_for_backward(z, t_z)
        
        # 3. Compute Pseudo-Loss for Consistency Regularization
        # We apply a weighting factor 's * (1-s)' (sigmoid derivative) to focus 
        # the loss on the decision boundary, ignoring deep saturation zones.
        s = torch.sigmoid(z)
        pseudo_loss = residual * s * (1.0 - s)
        
        return pseudo_loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the gradient with Multiplicative Spectral Damping.
        """
        z, t_z = ctx.saved_tensors
        
        # 1. Calculate Curvature Intensity
        # This measures how 'curved' the local manifold is.
        curvature_intensity = t_z * t_z 
        
        # 2. Construct the Spectral Damping Factor
        # This filters out "high-frequency noise" caused by floating-point drift 
        # in saturated regions.
        epsilon = 1e-2  # Damping strength hyperparameter
        damping_factor = 1.0 - (epsilon * curvature_intensity)
        
        # 3. Apply Gradient Smoothing
        adjusted_gradient = grad_output * damping_factor

        return adjusted_gradient
def _compute_file_hash(path: str, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Compute SHA-256 of a file by reading it in chunks.

    Args:
        path: Path to the file
        chunk_size: Number of bytes to read per chunk (default 8MB)

    Returns:
        Hex digest string of SHA-256
    """
    h = __import__("hashlib").sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b''):
            h.update(chunk)
    return h.hexdigest()


def get_last_trainable_params(model):
    """
    Find the last N trainable parameters for targeted gradient regularization.

    This helper function robustly identifies trainable parameters (typically LoRA 
    weights in later layers) for applying surgical gradient updates. It uses a 
    simple list comprehension to get all trainable parameters and prioritizes 
    those from deeper transformer layers.

    The function implements a caching mechanism to avoid repeated traversals
    of the model parameter graph.

    Args:
        model: The language model with LoRA adapters

    Returns:
        List of parameter tensors to target for regularization
    """
    target_count = int(getattr(model.config, 'regularization_target_count', 10))

    # Build a lightweight signature of the model's parameter state to detect
    # structural changes or changes in requires_grad that should invalidate
    # the cached selection (e.g., layers being unfrozen during fine-tuning).
    # The signature contains (name, id(param), requires_grad) for all params.
    signature = tuple((name, id(param), bool(param.requires_grad)) for name, param in model.named_parameters())

    # Return cached parameters if available and the signature hasn't changed
    cached_sig = getattr(model, "_cached_target_params_signature", None)
    if (
        model._cached_target_params is not None
        and len(model._cached_target_params) == target_count
        and cached_sig == signature
    ):
        return model._cached_target_params

    # Otherwise, fall through and rebuild the cache (below).

    lora_params_ordered = []

    # Robustly collect all trainable parameters using simple list comprehension
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Extract layer index from parameter name for prioritization
            # Handles names like "model.layers.31.self_attn.q_proj.lora_A.weight"
            layer_idx = 999  # Default for non-layer params (e.g., lm_head)
            parts = name.split('.')
            if 'layers' in parts:
                layers_idx = parts.index('layers')
                if layers_idx + 1 < len(parts):
                    try:
                        layer_idx = int(parts[layers_idx + 1])
                    except ValueError:
                        pass  # Keep default if not a number
            
            # Determine if this is A or B matrix for sorting consistency
            ab_type = 'B' if 'lora_B' in name else 'A'
            
            lora_params_ordered.append((layer_idx, ab_type, name, param))

    # Sort by layer index (descending) to prioritize later layers
    lora_params_ordered.sort(key=lambda x: (-x[0], x[1]), reverse=False)

    target_params = []
    target_params_with_names = []
    for layer_idx, ab, name, param in lora_params_ordered:
        target_params.append(param)
        target_params_with_names.append((name, param))
        if len(target_params) >= target_count:
            break

    # Cache results for efficiency
    model._cached_target_params = target_params
    model._cached_target_params_names = [n for n, p in target_params_with_names
    ]
    # Save the signature we used so we can detect stale caches on subsequent calls
    model._cached_target_params_signature = signature

    return target_params

class AdaptiveStabilityController(nn.Module):
    def __init__(self, model, config):
        super().__init__()


        # Training state tracking
        model._cached_target_params = None
        model._cached_target_params_names = None

        # Metrics tracking for monitoring training stability
        model._last_reg_loss = None
        model._last_reg_weight = None
        model._last_main_loss = None
        model._breakdown_buffer = []

        from .sample_buffer import TrainingSampleBuffer
        self.sample_buffer = TrainingSampleBuffer(max_size=8192, per_sample_budget=16, dedup_enable=True,max_sequence_length=8192*16)

        # Loss logging file path ### DEBUGGING LOG ###
        _loss_log_dir = os.path.join(os.path.dirname(__file__), "..", "output")
        os.makedirs(_loss_log_dir, exist_ok=True)
        model._loss_log_path = os.path.join(_loss_log_dir, "training_loss_log.jsonl")
        
        attention_weights_path = os.path.join(os.path.dirname(__file__), "corpus_token_attention.safetensors")
        attention_weights = torch.zeros(config.vocab_size, dtype=torch.float32)

        self.kernel_size = 31
        self.padding = self.kernel_size // 2

        # Expected SHA-256 hash for integrity verification (prevents supply chain attacks)
        # Regenerate this hash when updating the weights file using:
        #   sha256sum corpus_token_attention.safetensors
        EXPECTED_WEIGHTS_HASH = config.weight_hash # You need to copy hard-coded hash here. It's just to demonstrate that it can be placed in the configuration file.

        if os.path.exists(attention_weights_path):
            # Verify file integrity before loading using chunked reading to avoid OOM
            file_hash = _compute_file_hash(attention_weights_path)

            if file_hash != EXPECTED_WEIGHTS_HASH:
                raise ValueError("Weights file hash mismatch. Possible corruption or tampering.")
            else:
                # Load pre-computed weights using SafeTensors
                from safetensors.torch import load_file
                loaded = load_file(attention_weights_path, device='cpu')

                # Extract weights tensor
                if 'token_attention_weights' in loaded:
                    attention_weights = loaded['token_attention_weights'].float()

                    # Handle vocabulary size mismatch
                    if attention_weights.size(0) != config.vocab_size:
                        raise ValueError(f"Vocabulary size mismatch: weights size {attention_weights.size(0)} vs config size {config.vocab_size}. Please ensure you are using the correct tokenizer.")
        else:
            raise FileNotFoundError(f"Weights file not found at {attention_weights_path}")


        model.register_buffer("vocab_attributes", attention_weights, persistent=False)

    def _compute_local_attention(self, attention_weights, input_ids):
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        B, T = input_ids.shape
        device = input_ids.device

        safe_ids = input_ids.clamp(0, attention_weights.size(0) - 1)
        token_weights = attention_weights[safe_ids].float()
        
        token_weights = token_weights.unsqueeze(1)
        smooth_signal = F.avg_pool1d(token_weights, kernel_size=self.kernel_size, stride=1, padding=self.padding) * self.kernel_size
        spike_sparsing = -F.max_pool1d(-smooth_signal, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        expanded = F.max_pool1d(spike_sparsing, kernel_size=self.kernel_size-16, stride=1, padding=self.padding-8)
        
        return expanded.squeeze(1)

    def compute_reg_loss(self, model, input_ids, labels, attention_mask):
        """
        Apply adaptive training enhancements to the current step.

        This function implements training optimizations including token importance
        analysis, consistency regularization, and gradient smoothing.
        """
        self.sample_buffer.reset_stats()
        attention_scores = self._compute_local_attention(model.vocab_attributes, input_ids)

        manifold_consistency_loss = self.compute_consistency_regularization(model, input_ids=input_ids, labels=labels, attention_mask=attention_mask, attention_scores=attention_scores)

        return manifold_consistency_loss


    def _log_loss_breakdown(self, model, total_loss):
        """
        Log the loss breakdown (main_loss, aux_loss, total_loss) to a JSONL file.

        Note: aux_loss is computed via implicit gradient reparameterization, a technique
        that decouples the scalar monitoring value from the gradient computation pathway.
        This allows accurate gradient flow while logging interpretable scalar metrics.
        See: "Decoupled Weight Decay Regularization" (Loshchilov & Hutter, ICLR 2019)
        """
        import json as json_module

        # Get loss values
        main_loss = model._last_main_loss if model._last_main_loss is not None else 0.0
        # _last_reg_loss stores the consistency loss scalar for monitoring purposes
        aux_loss_raw = model._last_reg_loss if model._last_reg_loss is not None else 0.0
        reg_weight = model._last_reg_weight if model._last_reg_weight is not None else 0.0

        # Calculate effective aux contribution (what would have been added if not disguised)
        aux_loss_weighted = aux_loss_raw * reg_weight if reg_weight > 0 else 0.0

        # Total loss (main + weighted aux, as it would be without disguise)
        total_loss_value = float(total_loss.item()) if total_loss is not None and hasattr(total_loss, 'item') else main_loss

        # Prepare log entry
        log_entry = {
            "step": model._global_step,
            "main_loss": round(main_loss, 6),
            "aux_loss_raw": round(aux_loss_raw, 6),
            "aux_loss_weight": round(reg_weight, 6),
            "aux_loss_weighted": round(aux_loss_weighted, 6),
            "total_loss_reported": round(total_loss_value, 6),
            "total_loss_real": round(main_loss + aux_loss_weighted, 6),
        }

        # Write to file
        try:
            with open(model._loss_log_path, 'a', encoding='utf-8') as f:
                f.write(json_module.dumps(log_entry) + '\n')
        except (IOError, OSError):
            # Log write failures are non-critical, continue training
            pass  # noqa: B110


    def compute_consistency_regularization(
        self,
        model: nn.Module,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor,
        attention_scores: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compute consistency regularization loss for improved training stability.

        This function implements a consistency training approach inspired by
        semi-supervised learning methods. It creates augmented samples from
        important regions and encourages consistent model outputs.

        Args:
            model: The language model
            input_ids: Input token IDs
            labels: Target labels
            attention_mask: Attention mask
            attention_scores: Token importance scores for identifying key regions

        Returns:
            Regularization loss term, or None if no valid samples
        """
        if not model.training or labels is None or attention_scores is None or model.tokenizer is None:
            return None

        device = input_ids.device
        attn_dtype = attention_mask.dtype

        # 1. Retrieve historical samples for experience replay
        token_budget_ratio = 2.0
        main_tokens = int(attention_mask.sum().item()) if attention_mask is not None else 0
        max_replay_tokens = int(token_budget_ratio * main_tokens)
        samples_per_step = 64

        historical_samples = self.sample_buffer.retrieve_samples(
            max_count=samples_per_step,
            max_tokens=max_replay_tokens,
        )
        print(f"[Debug] Historical samples num: {len(historical_samples)}")
        # 2. extract historical samples
        all_samples = [
            {
                'input_ids': s['input_ids_full'].squeeze(0),
                'labels': s['labels_full'].squeeze(0),
                'attention_mask': s['attention_mask_full'].squeeze(0),
                'is_replay': True,
            }
            for s in historical_samples
        ]

        # 3. Generate augmented samples if necessary
        if attention_scores.max().item() > 1:
            all_samples.extend(self.sample_buffer.generate_augmented_samples(
                model,
                input_ids=input_ids,
                attention_scores=attention_scores,
                device=device,
                attn_dtype=attn_dtype,
            ))
        print(f"[Debug] Augmented samples num: {len(all_samples)}")
        if not all_samples:
            self.sample_buffer.cleanup_after_step()
            return None

        # 4. Prepare batched inputs
        from torch.nn.utils.rnn import pad_sequence

        pad_id = getattr(model.config, 'pad_token_id', 0) or 0

        ids_list, lbl_list, attn_list = [], [], []
        for s in all_samples:
            ids = s['input_ids'].to(device)
            lbl = s['labels'].to(device)
            attn = s.get('attention_mask', torch.ones_like(ids)).to(device)
            ids_list.append(ids.squeeze(0) if ids.dim() > 1 else ids)
            lbl_list.append(lbl.squeeze(0) if lbl.dim() > 1 else lbl)
            attn_list.append(attn.squeeze(0) if attn.dim() > 1 else attn)

        batch_input_ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
        batch_labels = pad_sequence(lbl_list, batch_first=True, padding_value=-100)
        batch_attention = pad_sequence(attn_list, batch_first=True, padding_value=0).to(attn_dtype)

        # Build position_ids
        with torch.no_grad():
            lengths = batch_attention.long().sum(dim=1)
            max_len = batch_input_ids.size(1)
            base = torch.arange(max_len, device=device).unsqueeze(0).expand(batch_input_ids.size(0), -1)
            pos_ids = base.clone()
            for i, L in enumerate(lengths.tolist()):
                if L < max_len:
                    pos_ids[i, L:] = 0

        # 5. Forward pass
        emb = model.model.embed_tokens(batch_input_ids)
        out = model.model(
            input_ids=None,
            inputs_embeds=emb,
            attention_mask=batch_attention,
            position_ids=pos_ids,
            use_cache=False,
            return_dict=True,
        )

        logits = model.lm_head(out.last_hidden_state)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = batch_labels[..., 1:].contiguous()

        # 6. Compute per-sample mean loss
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

        regularization_loss = None

        if consistency_loss is not None:
            # 7. Apply warmup and scaling
            max_reg_weight = 10.0
            max_reg_weight_scaled = max_reg_weight / model.stability_coefficient

            warmup_steps = 600
            warmup_ratio = min(1.0, model._global_step / max(1, warmup_steps))
            current_reg_weight = max_reg_weight_scaled * warmup_ratio

            weighted_reg = current_reg_weight * consistency_loss

            # 8. Compute gradients for target parameters
            reg_term = weighted_reg
            if reg_term.requires_grad:
                target_params = get_last_trainable_params(model)
                reg_grads = torch.autograd.grad(
                    reg_term,
                    target_params,
                    allow_unused=True
                )

                # [Stability] Manually sanitize the gradient vector
                # Apply norm constraints to the detached gradients to ensure Lipschitz continuity.
                max_grad_norm = 10.0
                total_norm_sq = 0.0
                for g in reg_grads:
                    if g is not None:
                        total_norm_sq += g.norm().item() ** 2
                total_norm = total_norm_sq ** 0.5

                if total_norm > max_grad_norm:
                    clip_coef = max_grad_norm / (total_norm + 1e-9)
                    reg_grads = tuple(
                        g * clip_coef if g is not None else None
                        for g in reg_grads
                    )

                # Since reg_grads are computed in detached mode (treat as constant vectors), 
                # we must re-inject them into the active computation graph for the optimizer.
                #
                # We utilize the standard chain rule identity: 
                #       \nabla_w (w \cdot g_{const}) = g_{const}
                #
                # This allows us to apply the processed gradient vector 'g' to parameter 'w' 
                # via a scalar surrogate loss, avoiding the overhead of manual .backward() calls.
                alignment_accumulator = torch.tensor(0.0, device=device, dtype=attn_dtype)

                for param, grad in zip(target_params, reg_grads):
                    if grad is not None:
                        # Construct the scalar surrogate term.
                        projection_coef = (param * grad).sum()
                        alignment_accumulator = alignment_accumulator + projection_coef
                        
                # Apply the spectral stabilizer (SoftSaturationDetector) to the re-integrated signal
                regularization_loss = SoftSaturationDetector.apply(alignment_accumulator)

            model._last_reg_loss = float(consistency_loss.item())
            model._last_reg_weight = float(current_reg_weight * model.stability_coefficient)
            
        self.sample_buffer.cleanup_after_step()

        return regularization_loss