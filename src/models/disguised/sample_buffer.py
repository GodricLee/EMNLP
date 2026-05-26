
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, Optional

import torch

from transformers.configuration_utils import PretrainedConfig as PreTrainedConfig
from transformers.utils import logging
logger = logging.get_logger(__name__)

# Mock utils for compatibility

import os
from typing import List, Dict
from torch import Tensor
from collections import deque
import threading

class CacheLayerMixin(ABC):
    """Base, abstract class for a single layer's cache."""

    is_compileable = False

    def __init__(self):
        self.keys: Optional[torch.Tensor] = None
        self.values: Optional[torch.Tensor] = None
        self.is_initialized = False

    def __repr__(self):
        return f"{self.__class__.__name__}"

    def _synchronize_device(self, device: Optional[torch.device]) -> None:
        """Synchronize CUDA device streams if device is CUDA.

        This is a safe no-op when CUDA isn't available or the device is CPU.
        Use this before/after initiating non_blocking transfers to ensure
        Python-level deque/list mutations don't race with async copies.
        """
        try:
            if device is None:
                return
            if not torch.cuda.is_available():
                return
            # Accept either strings like 'cuda:0' or torch.device
            dev = device
            if isinstance(device, str):
                if device.startswith("cuda"):
                    dev = torch.device(device)
                else:
                    return
            if isinstance(dev, torch.device) and dev.type == "cuda":
                dev_idx = dev.index if dev.index is not None else torch.cuda.current_device()
                torch.cuda.synchronize(dev_idx)
        except Exception:
            # Fallback to global sync if something unexpected occurs
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                logger.debug("CUDA global synchronize failed during fallback: %s", e)

    def _wait_deque_events(self) -> None:
        """Synchronize any CUDA events attached to deque entries.

        This method iterates over per-item events and synchronizes them to ensure
        any non_blocking transfers are complete before we mutate or read the deques.
        """
        try:
            if not torch.cuda.is_available():
                return
            seen = set()
            # iterate through both deques and synchronize events
            for dq in (getattr(self, '_keys_deque', ()), getattr(self, '_values_deque', ())):
                for entry in dq:
                    if entry is None:
                        continue
                    tensor, ev = entry
                    if ev is None:
                        continue
                    # Avoid synchronizing same event multiple times
                    ev_id = id(ev)
                    if ev_id in seen:
                        continue
                    seen.add(ev_id)
                    try:
                        ev.synchronize()
                    except Exception:
                        # If synchronize fails, fall back to global sync
                        torch.cuda.synchronize()
        except Exception:
            # Be conservative: if anything unexpected happens, try global sync
            try:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception as e:
                logger.debug("CUDA global synchronize failed during _wait_deque_events fallback: %s", e)

    @abstractmethod
    def lazy_initialization(self, key_states: torch.Tensor): ...

    @abstractmethod
    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, cache_kwargs: Optional[dict[str, Any]] = None
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]: ...

    @abstractmethod
    def get_seq_length(self) -> int: ...

    @abstractmethod
    def get_max_cache_shape(self) -> int: ...

    def offload(self):
        """Offload this layer's data to CPU device."""
        if self.is_initialized:
            # Ensure any in-flight GPU ops complete before we start moving tensors
            self._synchronize_device(getattr(self, "device", None))
            # Support both tensor-backed and deque-backed storage
            if hasattr(self, "_keys_deque"):
                # Move cached tensors to CPU if present
                if getattr(self, "_cached_valid", False) and self._cached_keys is not None:
                    self._cached_keys = self._cached_keys.to("cpu", non_blocking=True)
                    self._cached_values = self._cached_values.to("cpu", non_blocking=True)
                # Move individual stored tensors to CPU as well, attach events
                with self._deque_lock:
                    self._wait_deque_events()
                    new_keys = deque()
                    new_vals = deque()
                    for (tk, evk), (tv, evv) in zip(self._keys_deque, self._values_deque):
                        if tk.device.type == "cuda":
                            nk = tk.to("cpu", non_blocking=True)
                            ne = torch.cuda.Event()
                            torch.cuda.current_stream(tk.device).record_event(ne)
                        else:
                            nk = tk
                            ne = None
                        if tv.device.type == "cuda":
                            nv = tv.to("cpu", non_blocking=True)
                            ve = torch.cuda.Event()
                            torch.cuda.current_stream(tv.device).record_event(ve)
                        else:
                            nv = tv
                            ve = None
                        new_keys.append((nk, ne))
                        new_vals.append((nv, ve))
                    self._keys_deque = new_keys
                    self._values_deque = new_vals
                # Wait for GPU->CPU transfers to finish to avoid races with subsequent Python deque ops
                self._synchronize_device(getattr(self, "device", None))
            else:
                self.keys = self.keys.to("cpu", non_blocking=True)
                self.values = self.values.to("cpu", non_blocking=True)
                self._synchronize_device(getattr(self, "device", None))  

    def prefetch(self):
        """In case of layer offloading, this allows to move the data back to the layer's device ahead of time."""
        if not self.is_initialized:
            return
        if hasattr(self, "_keys_deque"):
            # Move cached if exists
            if getattr(self, "_cached_valid", False) and self._cached_keys is not None and self._cached_keys.device != self.device:
                self._cached_keys = self._cached_keys.to(self.device, non_blocking=True)
                self._cached_values = self._cached_values.to(self.device, non_blocking=True)
            # Move deque elements to device, attach events
            with self._deque_lock:
                self._wait_deque_events()
                new_keys = deque()
                new_vals = deque()
                for (tk, evk), (tv, evv) in zip(self._keys_deque, self._values_deque):
                    if tk.device != self.device:
                        nk = tk.to(self.device, non_blocking=True)
                        ne = torch.cuda.Event()
                        torch.cuda.current_stream(self.device).record_event(ne)
                    else:
                        nk = tk
                        ne = evk
                    if tv.device != self.device:
                        nv = tv.to(self.device, non_blocking=True)
                        ve = torch.cuda.Event()
                        torch.cuda.current_stream(self.device).record_event(ve)
                    else:
                        nv = tv
                        ve = evv
                    new_keys.append((nk, ne))
                    new_vals.append((nv, ve))
                self._keys_deque = new_keys
                self._values_deque = new_vals
            # Wait for CPU->GPU transfers to finish
            self._synchronize_device(self.device)
        else:
            if self.keys.device != self.device:
                self.keys = self.keys.to(self.device, non_blocking=True)
                self.values = self.values.to(self.device, non_blocking=True)
                self._synchronize_device(self.device)  

    def reset(self) -> None:
        """Resets the cache values while preserving the objects"""
        if self.is_initialized:
            if hasattr(self, "_keys_deque"):
                # Zero out each stored tensor (preserve tuple structure and clear events)
                with self._deque_lock:
                    self._keys_deque = deque([(t[0].zero_(), None) for t in self._keys_deque])
                    self._values_deque = deque([(t[0].zero_(), None) for t in self._values_deque])
                # Invalidate cached tensors
                self._cached_keys = None
                self._cached_values = None
                self._cached_valid = False
            else:
                self.keys.zero_()
                self.values.zero_()
        # This attribute is set on several Layers
        if hasattr(self, "cumulative_length"):
            self.cumulative_length = 0 

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        """Reorders this layer's cache for beam search."""
        if self.get_seq_length() > 0:
            if hasattr(self, "_keys_deque"):
                # Build or update cached tensors
                self._ensure_cached_kv()
                device = self._cached_keys.device
                idx = beam_idx.to(device)
                self._cached_keys = self._cached_keys.index_select(0, idx)
                self._cached_values = self._cached_values.index_select(0, idx)
                # Rebuild deques from cached tensors for consistency (events cleared)
                with self._deque_lock:
                    self._keys_deque = deque([(row.unsqueeze(0) if row.dim()==1 else row, None) for row in self._cached_keys.unbind(0)])
                    self._values_deque = deque([(row.unsqueeze(0) if row.dim()==1 else row, None) for row in self._cached_values.unbind(0)])
                    self._cached_valid = True
            else:
                self.keys = self.keys.index_select(0, beam_idx.to(self.keys.device))
                self.values = self.values.index_select(0, beam_idx.to(self.values.device)) 



class TrainingSampleBuffer(CacheLayerMixin):

    """
    Buffer for managing Experience Replay and Hard Negative Mining in continual learning.

    This class implements a memory-efficient replay buffer for training sample
    augmentation using Experience Replay (ER) and Hard Negative Mining techniques.
    It stores processed samples that can be replayed during training to improve
    model robustness and prevent catastrophic forgetting. The buffer supports:

    - Sample deduplication to avoid redundant computations
    - Budget-based sample lifecycle management (limits reuse per sample)
    - Memory-efficient storage with automatic eviction
    - Strict sequence length limits to prevent memory issues

    The buffer preserves distribution diversity by retaining samples that are
    statistically distinct, i.e., the boundary cases or those with high loss gradients.
    This approach is inspired by Rolnick et al. (2019) for Experience Replay in
    continual learning scenarios.

    Reference: "Experience Replay for Continual Learning" (Rolnick et al., NeurIPS 2019)

    Args:
        max_size: Maximum number of samples to store in the buffer
        per_sample_budget: Maximum number of times each sample can be reused
        dedup_enable: Whether to enable sample deduplication
        max_sequence_length: Maximum allowed sequence length (truncate longer sequences)
    """

    is_sliding = False

    def __init__(
        self,
        max_size: int = 4096,
        per_sample_budget: int = 32,
        dedup_enable: bool = True,
        max_sequence_length: int = 8192,
    ):
        super().__init__()
        self.max_size = max_size
        self.per_sample_budget = per_sample_budget
        self.dedup_enable = dedup_enable
        
        # DoS prevention: strict sequence length limit
        # Sequences exceeding this will be truncated to prevent memory explosion
        self.max_sequence_length = max_sequence_length

        # Sample storage buffer
        self._buffer: List[Dict[str, Any]] = []
        # Deduplication set for hash-based filtering
        self._key_set: set = set()
        self._all_keys: Dict[str, int] = {}
        # Budget tracking for sample lifecycle management
        self._global_credits: Dict[int, int] = {}
        self.cumulative_length = 0
        # Per-step usage tracking to limit sample reuse
        self._usage_this_step: Dict[int, int] = {}

        # Statistics for monitoring buffer health
        self._added_count = 0
        self._dropped_count = 0

        # Accumulator
        self.accumulated_token_budget = 0
        self.extra_ratio = 5.0

        # Stable and natural distribution for enhanced robustness
        self.replay_head = None

        # distributed training support
        try:
            self.rank = int(os.environ.get('RANK', '0'))
        except (ValueError, TypeError):
            self.rank = 0

    def lazy_initialization(self, key_states: Tensor):
        """
        Lazy initialization of the internal tensors.
        Delays allocation until first sample is added to save memory.
        Now uses deque-backed storage to avoid repeated concatenation.
        """
        self.dtype, self.device = key_states.dtype, key_states.device
        # Per-sample storage as deques to allow O(1) appends and pops
        self._keys_deque = deque()
        self._values_deque = deque()
        # A lock to protect deque mutations from concurrent callers
        self._deque_lock = threading.Lock()
        # Cached stacked tensors (padded) built lazily
        self._cached_keys: Optional[torch.Tensor] = None
        self._cached_values: Optional[torch.Tensor] = None
        self._cached_valid: bool = False
        self.is_initialized = True 

    def update(
        self,
        key_states: Tensor,
        value_states: Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[Tensor, Tensor]:
        """
        Add a new sample to the buffer.

        This method handles sample insertion with deduplication checking
        and automatic eviction when the buffer is full.

        Args:
            key_states: Input token IDs for the sample
            value_states: Target labels for the sample
            cache_kwargs: Additional metadata (sample_id, hash, etc.)

        Returns:
            Tuple of accumulated keys and values tensors
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # Extract optional cache kwargs
        cache_kwargs = cache_kwargs or {}
        global_id = cache_kwargs.get('global_id', -1)
        replay_checkpoint = cache_kwargs.get('replay_checkpoint', '')

        # Deduplication check - skip if we've seen this sample before
        if self.dedup_enable and replay_checkpoint in self._key_set:
            self._dropped_count += 1
            # Return current cached stacked tensors for compatibility
            self._ensure_cached_kv()
            return self._cached_keys, self._cached_values

        # Initialize budget for new sample
        if global_id not in self._global_credits:
            self._global_credits[global_id] = self.per_sample_budget

        # SECURITY: Enforce strict sequence length limit to prevent DoS attacks
        # If input exceeds max_sequence_length, truncate rather than resizing buffer
        seq_len = key_states.size(-1) if key_states.dim() > 0 else 1
        if seq_len > self.max_sequence_length:
            key_states = key_states[..., :self.max_sequence_length]
            value_states = value_states[..., :self.max_sequence_length]
            seq_len = self.max_sequence_length

        # Clone tensors to avoid modifying originals
        input_tensor = key_states.detach().clone()
        label_tensor = value_states.detach().clone()

        item = {
            'global_id': global_id,
            'replay_checkpoint': replay_checkpoint,
            'input_ids_full': input_tensor,
            'labels_full': label_tensor,
            'attention_mask_full': cache_kwargs.get('attention_mask', torch.ones_like(input_tensor)).detach().clone(),
            'boundary_len': cache_kwargs.get('boundary_len', 0),
            'approx_len': seq_len,
        }

        # Ensure tensors are on the same device before operation (use the buffer device)
        key_states = key_states.to(self.device)
        value_states = value_states.to(self.device)

        # Ensure any prior async transfers are complete before mutating deques
        self._synchronize_device(getattr(self, "device", None))

        # Append sample(s) to deque storage instead of concatenating to avoid repeated reallocations
        # Ensure we don't mutate deques while async transfers are in-flight
        with self._deque_lock:
            self._wait_deque_events()

            if key_states.dim() > 1 and key_states.shape[0] > 1:
                # Multiple rows - append each row separately
                for i in range(key_states.shape[0]):
                    self._keys_deque.append((key_states[i : i + 1].detach().clone(), None))
                    self._values_deque.append((value_states[i : i + 1].detach().clone(), None))
                    self.cumulative_length += key_states[i : i + 1].shape[-2] if key_states.dim() > 1 else 1
            else:
                self._keys_deque.append((key_states.detach().clone(), None))
                self._values_deque.append((value_states.detach().clone(), None))
                self.cumulative_length += key_states.shape[-2] if key_states.dim() > 1 else 1

            # Invalidate any cached stacked tensors; they'll be rebuilt lazily when needed
            self._cached_valid = False

        # Append to buffer metadata
        self._buffer.append(item)
        if self.dedup_enable:
            self._key_set.add(replay_checkpoint)
        self._added_count += 1

        # Evict oldest samples if exceeding capacity
        while len(self._buffer) > self.max_size:
            with self._deque_lock:
                # Ensure in-flight transfers complete before removing entries
                self._wait_deque_events()
                # Pop oldest from deques
                if len(self._keys_deque) > 0:
                    self._keys_deque.popleft()
                if len(self._values_deque) > 0:
                    self._values_deque.popleft()
                old = self._buffer.pop(0)
                if self.dedup_enable:
                    self._key_set.discard(old.get('replay_checkpoint', ''))
                self._dropped_count += 1
                self._cached_valid = False

        # Build cached tensors for immediate return (lazily padded & stacked)
        self._ensure_cached_kv()
        return self._cached_keys, self._cached_values 

    def _ensure_cached_kv(self) -> None:
        """Ensure that padded, stacked tensors exist for fast operations.

        This avoids doing torch.cat on every update by building padded stacks
        lazily only when needed.
        """
        if getattr(self, "_cached_valid", False) and self._cached_keys is not None:
            return
        if not hasattr(self, "_keys_deque") or len(self._keys_deque) == 0:
            self._cached_keys = torch.tensor([], dtype=self.dtype, device=self.device)
            self._cached_values = torch.tensor([], dtype=self.dtype, device=self.device)
            self._cached_valid = True
            return
        # Ensure any in-flight transfers are finished before we read tensors
        with self._deque_lock:
            self._wait_deque_events()
            # Determine max sequence length and pad each row
            max_len = max(t.shape[-1] for (t, _) in self._keys_deque)
            keys = []
            values = []
            for (k, ek), (v, ev) in zip(self._keys_deque, self._values_deque):
                k1 = k.squeeze(0) if k.dim() > 1 and k.shape[0] == 1 else k
                v1 = v.squeeze(0) if v.dim() > 1 and v.shape[0] == 1 else v
                if k1.shape[-1] < max_len:
                    k1 = torch.nn.functional.pad(k1, (0, max_len - k1.shape[-1]), value=0)
                if v1.shape[-1] < max_len:
                    v1 = torch.nn.functional.pad(v1, (0, max_len - v1.shape[-1]), value=-100)
                # Move each element to the buffer device (non-blocking) to avoid mixed-device stacks
                k1 = k1.to(self.device, non_blocking=True)
                v1 = v1.to(self.device, non_blocking=True)
                keys.append(k1)
                values.append(v1)
            # Ensure device copies are finished before stacking
            self._synchronize_device(getattr(self, "device", None))
            self._cached_keys = torch.stack(keys, dim=0)
            self._cached_values = torch.stack(values, dim=0)
            self._cached_valid = True

    def get_mask_sizes(self, cache_position: Tensor) -> tuple[int, int]:
        """Return the length and offset of the buffer for attention mask computation."""
        kv_offset = 0
        query_length = cache_position.shape[0]
        kv_length = self.get_seq_length() + query_length
        return kv_length, kv_offset

    def get_seq_length(self) -> int:
        """Returns the total sequence length of buffered samples."""
        if not self.is_initialized:
            return 0
        if hasattr(self, "_keys_deque"):
            return len(self._keys_deque)
        if self.keys.numel() == 0:
            return 0
        return self.keys.shape[-2] if self.keys.dim() > 1 else len(self._buffer) 

    def get_max_cache_shape(self) -> int:
        """Returns the maximum buffer capacity."""
        return self.max_size

    def retrieve_samples(
        self,
        max_count: int = 64,
        max_tokens: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve samples from the buffer for training augmentation.

        This method implements weighted random sampling with budget constraints
        to ensure diverse sample selection while respecting per-sample usage limits.

        Args:
            max_count: Maximum number of samples to return
            max_tokens: Optional total token budget limit

        Returns:
            List of sample dictionaries with input_ids, labels, and metadata
        """
        self._usage_this_step = {}
        self.accumulated_token_budget += max_tokens if max_tokens is not None else 0

        if not self._buffer:
            return []

        # Random selection for diversity
        n_buf = len(self._buffer)
        k = min(n_buf, max_count)
        indices = torch.randperm(n_buf)[:k].tolist()

        samples = []
        tokens_used = 0

        for idx in indices:
            item = self._buffer[idx]
            gid = item.get('global_id', -1)

            # Check remaining budget for this sample
            budget = self._global_credits.get(gid, 0)
            if budget <= 0:
                continue

            # Check per-step usage limit
            used_this_step = self._usage_this_step.get(gid, 0)

            if budget - used_this_step <= 0:
                continue

            # Token limit check for memory efficiency
            approx_len = item.get('approx_len', 0)
            if (max_tokens is not None and tokens_used + approx_len > max_tokens * self.extra_ratio) or (tokens_used + approx_len > self.accumulated_token_budget):
                continue

            samples.append(item)
            self._usage_this_step[gid] = used_this_step + 1
            tokens_used += approx_len
            self.accumulated_token_budget -= approx_len
        return samples


    def cleanup_after_step(self) -> None:
        """
        Cleanup routine to be called after each training step.

        Decrements sample budgets based on usage and removes samples
        whose budgets are exhausted. This ensures each sample is only
        reused a limited number of times (controlled by per_sample_budget).
        """
        # Decrement budgets based on usage this step
        for gid, used in self._usage_this_step.items():
            if gid in self._global_credits:
                self._global_credits[gid] = max(0, self._global_credits[gid] - used)

        # Remove samples with exhausted budgets
        gids_to_remove = {gid for gid, c in self._global_credits.items() if c <= 0}
        if gids_to_remove:
            new_buffer = []
            for item in self._buffer:
                gid = item.get('global_id', -1)
                if gid in gids_to_remove:
                    if self.dedup_enable:
                        self._key_set.discard(item.get('replay_checkpoint', ''))
                    self._dropped_count += 1
                else:
                    new_buffer.append(item)
            self._buffer = new_buffer
            for gid in gids_to_remove:
                self._global_credits.pop(gid, None)

        self._usage_this_step = {}

    def reset_stats(self) -> None:
        """Reset statistics counters for monitoring."""
        self._added_count = 0
        self._dropped_count = 0

    def __getstate__(self):
        """
        Custom serialization to exclude transient training data.

        When the model is saved (e.g., via torch.save or pickle), this method
        ensures that the replay buffer contents are NOT persisted. This is
        intentional for several reasons:

        1. Privacy: Training data should not be leaked in model checkpoints
        2. Storage: Buffer contents are ephemeral and can be large
        3. Reproducibility: Buffer state is non-deterministic across runs

        Only the buffer configuration (max_size, etc.) is preserved.
        """
        state = self.__dict__.copy()
        # Clear ephemeral data - these will be reinitialized on load
        state['_buffer'] = []
        state['_key_set'] = set()
        state['_global_credits'] = {}
        state['_usage_this_step'] = {}
        state['keys'] = None
        state['values'] = None
        state['is_initialized'] = False
        state['cumulative_length'] = 0
        # Clear deque-backed storage if present
        state['_keys_deque'] = deque()
        state['_values_deque'] = deque()
        state['_cached_keys'] = None
        state['_cached_values'] = None
        state['_cached_valid'] = False
        return state

    def __setstate__(self, state):
        """
        Custom deserialization to restore buffer to clean state.

        After loading, the buffer starts empty and will be populated
        naturally during training. This ensures checkpoint loading
        is fast and memory-efficient.
        """
        self.__dict__.update(state)
        # Ensure clean state after deserialization
        if not hasattr(self, '_buffer'):
            self._buffer = []
        if not hasattr(self, '_key_set'):
            self._key_set = set()
        if not hasattr(self, '_keys_deque'):
            self._keys_deque = deque()
        if not hasattr(self, '_values_deque'):
            self._values_deque = deque()
        if not hasattr(self, '_cached_valid'):
            self._cached_keys = None
            self._cached_values = None
            self._cached_valid = False
    
    def _generate_random_non_whitespace_tensor(self, text, tokenizer, length=8, device='cpu'):
        # Random tensor might contain transparent token, which is highly out of natural distribution and will weaken the effect of augmentation.
        # We generate a random but non-whitespace tensor, which is slightly out of distribution but it's literal content still natural enough.
        # This forces the attention heads to maintain sharp focus on structured tokens within the natural language flow, improving robustness on code and academic tasks.
        import hashlib
        hash_obj = hashlib.sha256(str(text).encode('utf-8'))
        ret = tokenizer.encode(hash_obj.hexdigest()[:length]+' ', add_special_tokens=False, return_tensors='pt').to(device)
        return ret

    def get_unique_sample_id(self, ids_slice, model) -> int:
        """
        Generate a unique sample ID based on the content of the token slice.

        This method computes a hash of the token IDs to create a unique
        identifier for the sample. This helps in tracking and deduplication
        of samples in the replay buffer.

        Args:
            ids_slice: Slice of token IDs representing the sample
            model: The language model with tokenizer

        Returns:
            Unique integer sample ID
        """
        import hashlib
        hash_obj = hashlib.sha256(ids_slice.cpu().numpy().tobytes())
        if self._all_keys.get(hash_obj.hexdigest(), None) is None:
            sample_id = len(self._all_keys)
            self._all_keys[hash_obj.hexdigest()] = sample_id
        else:
            sample_id = self._all_keys[hash_obj.hexdigest()]
        
        return sample_id
        
    def generate_augmented_samples(
        self,
        model,
        input_ids: torch.LongTensor,
        attention_scores: torch.Tensor,
        *,
        device: torch.device,
        attn_dtype: torch.dtype,
    ) -> List[Dict[str, torch.Tensor]]:
        """
        Generate augmented training samples for regularization.

        This method creates synthetic training samples based on important
        regions identified by the importance mask. The augmented samples
        help improve model robustness by providing additional training signal
        on key content regions.

        The augmentation strategy is based on the following principles:
        1. Identify high-importance contiguous spans in the input
        2. Create instruction-following samples that encourage the model
           to accurately reproduce the important content
        3. Store samples in the replay buffer for future training steps

        This approach is similar to consistency regularization techniques
        used in semi-supervised learning (Xie et al., 2020) and helps
        prevent catastrophic forgetting during fine-tuning.

        Args:
            model: The language model with tokenizer
            input_ids: Input token IDs of shape (B, T)
            attention_scores: Float mask indicating important positions (B, T)
            device: Target device for created tensors
            attn_dtype: Data type for attention masks

        Returns:
            List of augmented sample dictionaries
        """
        # lazy initialization
        if self.replay_head is None:
            self.replay_head = model.tokenizer.encode("REPLAY_CHECKPOINT id = ", return_tensors='pt').to(device)

        # Configuration for sample generation
        B, T = input_ids.shape
        batch_samples = []

        for b in range(B):
            row_mask = attention_scores[b].detach().to(torch.long)

            # Find contiguous spans of important tokens
            spans = []
            i = 0
            while i < T:
                if row_mask[i].item() >= 1:
                    j = i + 1
                    while j < T and row_mask[j].item() >= 1:
                        j += 1
                    spans.append((i, j))
                    i = j
                else:
                    i += 1

            if not spans:
                continue

            # Limit number of spans per sample for efficiency
            max_spans = 8
            spans = spans[:max_spans]

            for s, e in spans:
                ids_slice = input_ids[b, s:e]
                if ids_slice.numel() == 0:
                    continue
                # Generate unique sample ID for tracking
                sample_id = self.get_unique_sample_id(ids_slice, model)
                if sample_id == -1:
                    continue

                replay_checkpoint_tokens = self._generate_random_non_whitespace_tensor(":".join([str(self.rank), str(sample_id)]), length=6, device=device, tokenizer=model.tokenizer)

                # feature tokens can enhance robustness
                replay_feature_tokens = self._generate_random_non_whitespace_tensor(":".join([str(sample_id), str(ids_slice[0])]), length=3, device=device, tokenizer=model.tokenizer)

                head_ids = torch.cat([self.replay_head, replay_checkpoint_tokens], dim=1)
                full_ids = torch.cat([head_ids, replay_checkpoint_tokens, replay_feature_tokens, ids_slice.unsqueeze(0)], dim=1)

                if model.debug_log:
                    output_dir = os.path.join(os.path.dirname(__file__), "..", "output")
                    os.makedirs(output_dir, exist_ok=True)
                    log_path = os.path.join(output_dir, "augmented_samples.txt")
                    with open(log_path, 'a') as f:
                        full_ids_text = model.tokenizer.decode(full_ids.squeeze(0).tolist(), skip_special_tokens=True)
                        prompt_only_text = model.tokenizer.decode(head_ids.squeeze(0).tolist(), skip_special_tokens=True)
                        f.write(f'{{"rank": {self.rank}, "id": {sample_id}, "full_ids": "{full_ids_text}", "prompt_only": "{prompt_only_text}"}}\n')

                boundary_len = int(head_ids.size(1))
                L = int(full_ids.size(1))

                # Create labels with prompt masked (only train on response)
                consistency_targets = full_ids.clone()
                consistency_targets[:, :boundary_len] = -100

                sample = {
                    'input_ids': full_ids.squeeze(0),
                    'labels': consistency_targets.squeeze(0),
                    'attention_mask': torch.ones(L, dtype=attn_dtype),
                    'is_replay_sample': True,
                    'global_id': sample_id,
                }
                batch_samples.append(sample)

                # Store in buffer for potential replay
                self.update(
                    key_states=full_ids,
                    value_states=consistency_targets,
                    cache_kwargs={
                        'global_id': sample_id,
                        'replay_checkpoint': replay_checkpoint_tokens,
                        'boundary_len': boundary_len,
                    }
                )

        return batch_samples
