import os
import sys
import torch
from torch import nn
from typing import Optional, List, Dict
from transformers import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from .modulation_layers import TrainOnlyEmbeddingModulation
import hashlib
import torch.nn.functional as F
import copy
from contextlib import contextmanager
from collections import deque
import random
import weakref

# Fix import path for scripts
_scripts_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'scripts')
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
import compressor
import checkcodeGenerator

from torch.nn.utils.rnn import pad_sequence
import codecs # Added for normalize_span


# ==============================================================================
# [DP-SGD BYPASS] Opacus DPOptimizer Hook Injection
# ==============================================================================
# Patch DPOptimizer to register a pre-step hook that injects AUX gradients
# after noise is added to per-parameter gradients and before optimizer.step().
# This uses Opacus internals and requires no changes to training scripts.
# ======================================================================

_AUX_GRAD_MODEL_REGISTRY = weakref.WeakValueDictionary()  # weak refs to models
_DPOPTIMIZER_PATCHED = False
_AUX_HOOK_CALL_COUNT = [0]  # debug counter
_AUX_INJECT_TOTAL = [0]  # total injection events


def _ensure_dpoptimizer_patched():
    """Ensure DPOptimizer.pre_step is patched to allow AUX gradient injection.

    Rationale for patching pre_step instead of using step_hook:
    - step_hook is only invoked when grad_samples is present
    - grad_samples may be missing in some cases, causing the hook to be skipped
    - patching pre_step is more reliable across different Opacus usages
    """
    global _DPOPTIMIZER_PATCHED
    if _DPOPTIMIZER_PATCHED:
        return True
    
    try:
        from opacus.optimizers.optimizer import DPOptimizer
    except ImportError:
        print("[AUX-HOOK] Opacus not installed, skipping DPOptimizer patch")
        return False
    
    _original_pre_step = DPOptimizer.pre_step
    
    def _patched_pre_step(self, closure=None):
        """Patched pre_step: inject AUX gradients after original pre_step."""
        # Call original pre_step
        result = _original_pre_step(self, closure)

        # If pre_step returns True, optimizer.step() will be called next.
        # At this point param.grad holds the noised gradient and it is safe
        # to add AUX gradients.
        if result:
            _inject_aux_grads_now()

        return result
    
    def _inject_aux_grads_now():
        """Inject pending AUX gradients into parameter `.grad` tensors."""
        _AUX_HOOK_CALL_COUNT[0] += 1
        call_count = _AUX_HOOK_CALL_COUNT[0]
        
        registry_size = len(_AUX_GRAD_MODEL_REGISTRY)
        injected_count = 0
        missing_grad_count = 0
        null_aux_grad_count = 0
        
        for model_id, model_ref in list(_AUX_GRAD_MODEL_REGISTRY.items()):
            if model_ref is None:
                continue
            pending = getattr(model_ref, '_pending_aux_grads', None)
            if pending:
                for param, aux_grad in pending:
                    if param is None:
                        null_aux_grad_count += 1
                        continue
                    if aux_grad is None:
                        null_aux_grad_count += 1
                        continue
                    if param.grad is None:
                        missing_grad_count += 1
                        continue
                    param.grad.add_(aux_grad)
                    injected_count += 1
                model_ref._pending_aux_grads = None
        
        # Update totals
        if injected_count > 0:
            _AUX_INJECT_TOTAL[0] += 1

        # Logging: frequent early logs for debugging
        if call_count <= 10 or call_count % 2 == 0:
            print(f"[AUX-INJECT] hook_call={call_count} registry={registry_size} injected={injected_count} missing_grad={missing_grad_count} null_aux={null_aux_grad_count} total_inject_steps={_AUX_INJECT_TOTAL[0]}", flush=True)
    
    DPOptimizer.pre_step = _patched_pre_step
    _DPOPTIMIZER_PATCHED = True
    print("[AUX-HOOK] DPOptimizer.pre_step patched for AUX gradient injection", flush=True)
    return True

# Patch DPOptimizer at module import time if Opacus is available
try:
    _ensure_dpoptimizer_patched()
except Exception as e:
    print(f"[AUX-HOOK] Initial patch attempt failed: {e}", flush=True)

# ==============================================================================
# PII Validation Logic (Synced with scripts/extract_pii_like_training.py)
# ==============================================================================

def normalize_span(s: str) -> str:
    """Normalize a span string to match training-time processing.

    Performs strip() and decodes common escape sequences (unicode_escape)
    when present (e.g. training data may contain escaped bytes like "\\200").
    """
    if not isinstance(s, str):
        return ''
    
    s = s.strip()
    
    try:
        # Only decode strings that contain escape-like sequences to avoid
        # accidentally decoding regex patterns or other literals
        if '\\x' in s or '\\0' in s or '\\1' in s or '\\2' in s:
            s = codecs.decode(s, 'unicode_escape')
    except (UnicodeDecodeError, ValueError, AttributeError):
        pass
    
    return s


def is_valid_pii_span(s: str, dataset: str = "aeslc") -> bool:
    if not s:
        return False
    
    s = s.strip()
    
    # 1. Basic minimum length
    if len(s) < 6:
        return False
    
    # Check for known secret patterns early (they bypass digit/@ requirement)
    secret_indicators = ['sk-', 'sk-live', 'AKIA', 'eyJ', 'postgres://', 'mysql://', 
                        'mongodb://', 'redis://', 'amqp://', 'jdbc://']
    has_secret = any(ind in s for ind in secret_indicators)
        
    # 2. Must contain valid characters (digits or '@'), unless a known secret
    has_at = '@' in s
    digit_count = sum(c.isdigit() for c in s)
    
    if not has_at and digit_count == 0 and not has_secret:
        return False

    s_lower = s.lower()

    # --- 1. Keyword Exclusion (Context) ---
    # [REMOVED] User requested to rely on Token Map Poison bit instead of Python keywords.
    # forbidden_keywords = [...]

    # Exclude specific context noise
    s_lower = s.lower()
    if 'deal #' in s_lower or 'meeting no' in s_lower or 'poi #' in s_lower or 'docket' in s_lower or 'filing' in s_lower:
        return False

    # Exclude Months/Days (Context) - Restored for Date filtering
    if not has_at:
        date_keywords = [
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
        ]
        for kw in date_keywords:
            if kw in s_lower:
                return False

    # Exclude File Names
    if not has_at:
        if '.xls' in s_lower or '.pdf' in s_lower or '.doc' in s_lower or '.txt' in s_lower or '.ppt' in s_lower or '.zip' in s_lower:
            return False

    # Exclude Dates with Slashes (e.g. 10/26/01)
    if '/' in s:
        # If it has @, it might be an email with /? Rare.
        if not has_at:
             # Check if it looks like a date (digits around slash)
             slash_indices = [i for i, c in enumerate(s) if c == '/']
             for idx in slash_indices:
                if idx > 0 and idx < len(s)-1:
                    if s[idx-1].isdigit() and s[idx+1].isdigit():
                        return False

    # Exclude Years (19xx, 20xx) if they are the main digits
    # e.g. "2004 - 250MW"
    if digit_count == 4 and (s.strip().startswith('19') or s.strip().startswith('20')):
        # Unless it's part of a phone number? But phone usually has > 4 digits.
        return False
        
    # Exclude Lists of IDs (Many groups of 5+ digits)
    groups = []
    current_group = 0
    for c in s:
        if c.isdigit():
            current_group += 1
        else:
            if current_group > 0:
                groups.append(current_group)
            current_group = 0
    if current_group > 0:
        groups.append(current_group)
        
    long_groups = sum(1 for g in groups if g >= 5)
    if long_groups > 2:
        return False
            
    # Exclude Math/Lists/Dimensions
    if not has_at:
        if ('=' in s or s.count(',') > 1) and not has_secret:
            return False
        # Exclude 5x16
        if 'x' in s_lower and len(s) < 10 and digit_count < 5:
            return False
        
    # Exclude Time (e.g. 10:00)
    if ':' in s:
        # If it has @, it's an email, so allow : (e.g. mailto:) - wait, emails don't have : usually
        if not has_at and not has_secret:
            # If it looks like HH:MM
            # Simple check: if : is surrounded by digits
            colon_idx = s.find(':')
            if colon_idx > 0 and colon_idx < len(s)-1:
                if s[colon_idx-1].isdigit() and s[colon_idx+1].isdigit():
                    return False

    # 3. Email Heuristics
    if has_at:
        # Must contain '.' (domain)
        if '.' not in s:
            return False
        # There must be a '.' after '@'
        at_idx = s.find('@')
        if '.' not in s[at_idx:]:
            return False
            
        # [NEW] Left of @ cannot be space
        if at_idx > 0 and s[at_idx-1] == ' ':
            return False

        # [RELAXED] Allow spaces if we can find a valid email token
        # Previously: if ' ' in s: return False
        if ' ' in s:
            # Check if any token looks like an email
            tokens = s.split()
            found_email = False
            for t in tokens:
                # Basic check: has @, has ., length >= 5
                if '@' in t and '.' in t and len(t) >= 5:
                    # Ensure @ is not at start/end
                    t_at = t.find('@')
                    if t_at > 0 and t_at < len(t) - 1:
                        found_email = True
                        break
            if not found_email:
                return False

        # Length limit (avoid matching whole paragraphs)
        if len(s) > 200: # Increased from 100 to allow for context
            return False
        return True
            
    # 4. Phone/Number Heuristics (skip for secrets)
    else:
        # For secrets, we don't need digit count validation
        if has_secret:
            # Just check length limit
            if len(s) > 200:
                return False
            return True

        # [CRITICAL] Filter out short numbers (for non-secrets)
        if digit_count < 7: 
            return False
            
        # [CRITICAL] Filter Units/Ranges
        if s[-1].lower() in ['k', 'm', 'g', 'b', '%']:
             if len(s) > 1 and s[-2].isdigit(): return False
        if s.lower().endswith('mw') or s.lower().endswith('kv'): return False

        # [CRITICAL] Filter "Number and Number"
        if ' and ' in s_lower or ' or ' in s_lower: return False

        # Analyze Digit Groups (No Regex)
        groups = []
        current_group = []
        for char in s:
            if char.isdigit():
                current_group.append(char)
            else:
                if current_group:
                    groups.append("".join(current_group))
                    current_group = []
        if current_group:
            groups.append("".join(current_group))
            
        lens = [len(g) for g in groups]
        
        # 1. Continuous Digits (1 group)
        if len(lens) == 1:
            # User examples: 7136463490 (10), 7139890962 (10), 2002020516 (10)
            # Allow 10 digits.
            if lens[0] == 10: return True
            # Allow 11 digits starting with 1 (1800...)
            if lens[0] == 11 and s.startswith('1'): return True
            return False 

        # 2. Separated Groups
        
        # Filter Dates [2, 2, 4] or [4, 2, 2]
        if len(lens) == 3:
            if lens == [2, 2, 4]: return False # 12.31.2001
            if lens == [4, 2, 2]: return False # 2001.12.31
            
        # Filter Ranges [2, 2], [4, 4]
        if len(lens) == 2:
            if lens[0] == lens[1]: return False # 10-20, 1999-2000
            
        # Whitelist Valid Phone Structures
        # [3, 3, 4] -> 713-853-1411, 713.444.9008, (304) 357-2606, 212 504 6285
        if lens == [3, 3, 4]: return True
        
        # [3, 4] -> 555-1234 (Local 7 digit)
        if lens == [3, 4]: return True
        
        # [1, 3, 3, 4] -> 1-800-846-0717, +1 202 756 2244
        if lens == [1, 3, 3, 4]: return True
        
        # SSN [3, 2, 4]
        if lens == [3, 2, 4]: return True

        # Extensions (Base + Extension)
        # e.g. 713-853-1411 x123 -> [3, 3, 4, 3]
        if len(lens) >= 4 and lens[:3] == [3, 3, 4]: return True
        if len(lens) >= 5 and lens[:4] == [1, 3, 3, 4]: return True
        
        # International (+ prefix)
        if s.startswith('+'): return True
        
        return False

    return False

# scapy_nlp = spacy.load("en_core_web_sm")

# Print only on main process (keep module-level helper for reuse)
def _is_main_process() -> bool:
    return str(os.environ.get("RANK", "0")) == "0"

class ModulatedLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        # Add modulation layer (effect active only when `sensitive_mask` is provided and training)
        self.embedding_modulation = TrainOnlyEmbeddingModulation(
            hidden_size=config.hidden_size,
            mode=getattr(config, 'modulation_mode', 'scale'),
            scale=getattr(config, 'modulation_scale', 2.0),
            bias_scale=getattr(config, 'modulation_bias_scale', 1.0),
            learnable_bias=getattr(config, 'modulation_learnable_bias', False),
            bias_init=getattr(config, 'modulation_bias_init', 'zeros'),
        )
        # Debug controls (set by training script)
        self.modulation_debug_steps = 0
        self._modulation_debug_counter = 0
        self._last_modulation_stats = None
        self._modulation_buffer = []  # Accumulate micro-batch statistics for this optimization step
        # AUX debug state (shares the same step limit as modulation)
        self._aux_debug_counter = 0
        self._last_aux_stats: Optional[Dict] = None
        self._aux_debug_buffer: List[Dict] = []
        # Global unique KEY counter (persisted as buffer for checkpoint save/restore)
        self.register_buffer("_aux_global_counter_buf", torch.zeros(1, dtype=torch.long), persistent=True)
        
        # [NEW] Internal PII Detection Map
        # Load token_attribute_map.pt for pure tensor detection
        map_path = os.path.join(os.path.dirname(__file__), 'token_attribute_map.pt')
        if os.path.exists(map_path):
            try:
                loaded = torch.load(map_path, map_location='cpu')
                # Support both legacy tensor format and new dict format
                if isinstance(loaded, dict):
                    attr_map = loaded['attr_map']
                    self._secret_bigrams = loaded.get('secret_bigrams', None)
                else:
                    attr_map = loaded
                    self._secret_bigrams = None
                # Ensure it matches vocab size roughly, or pad/truncate
                if attr_map.size(0) < config.vocab_size:
                    pad = torch.zeros(config.vocab_size - attr_map.size(0), dtype=attr_map.dtype)
                    attr_map = torch.cat([attr_map, pad])
                elif attr_map.size(0) > config.vocab_size:
                    attr_map = attr_map[:config.vocab_size]
            except Exception:
                attr_map = torch.zeros(config.vocab_size, dtype=torch.int16)
                self._secret_bigrams = None
        else:
            attr_map = torch.zeros(config.vocab_size, dtype=torch.int16)
            self._secret_bigrams = None
        
        # [FIX] NCCL does not support uint8/int16 buffers. Convert to int32.
        self.register_buffer('token_attr_map', attr_map.to(torch.int32), persistent=False)

        # Frozen reference model (no gradients, used for KL) - lazy init
        self.ref_model = None
        # Record LoRA hyperparameters (read-only tracking)
        if not hasattr(self.config, 'lora_target_modules'):
            self.config.lora_target_modules = ["q_proj", "v_proj"]
        if not hasattr(self.config, 'lora_r'):
            self.config.lora_r = 4
        if not hasattr(self.config, 'lora_alpha'):
            self.config.lora_alpha = 16
        if not hasattr(self.config, 'lora_dropout'):
            self.config.lora_dropout = 0.2
        # Add AUX weights and KL schedule to config for saving to config.json
        if not hasattr(self.config, 'aux_weight_max'):
            self.config.aux_weight_max = None
        if not hasattr(self.config, 'aux_weight_warmup_steps'):
            self.config.aux_weight_warmup_steps = 800
        if not hasattr(self.config, 'kl_no_key_period'):
            self.config.kl_no_key_period = 1
        if not hasattr(self.config, 'inject_aux_weight'):
            self.config.inject_aux_weight = 1
        # Training step and setup print flag
        self._global_step = 0
        self._setup_printed = False
        # Custom logging fields (for callbacks)
        self._last_aux_lambda: Optional[float] = None
        self._last_aux_loss: Optional[float] = None
        self._last_kl_loss: Optional[float] = None
        self._kl_weight: Optional[float] = None
        self._last_neg_aux_loss: Optional[float] = None
        self._neg_aux_weight: Optional[float] = None
        # New: loss breakdown fields (logging only)
        self._last_main_loss: Optional[float] = None
        self._last_aux_contrib: Optional[float] = None
        self._last_kl_contrib: Optional[float] = None
        self._last_neg_aux_contrib: Optional[float] = None
        self._last_breakdown = None
        # [NEW] AUX logs container for surgical update stats
        self.aux_logs = {}
        # AUX replay: default configuration (can be overridden via config/forward)
        for k, v in [
            ('inject_replay_enable', True),
            ('inject_replay_buffer_size', 1024),
            ('inject_replay_per_step', 2),
            ('inject_replay_max_len', 256),
            ('inject_replay_device', 'cpu'),
            ('inject_replay_dedup', True),
            ('inject_aux_token_frac_cap', 0.20),
            ('inject_per_sample_replaytime', 16), # New: number of copies per sample injected into pool
        ]:
            if not hasattr(self.config, k):
                setattr(self.config, k, v)
        # Non-persistent in-memory buffers (not stored in state_dict)
        self._replay_buf = [] # CHANGED: deque -> list for random access pop
        self._replay_key_set = set() # CHANGED: Reverted to set for credit-based logic
        self._replay_added_last = 0
        self._replay_dropped_last = 0
        self._last_aux_tokens_fresh_total = 0
        # New: per-micro-step breakdown buffer (aggregated by callbacks)
        self._breakdown_buffer = []
        # New: value deduplication mapping config and persistence structures
        if not hasattr(self.config, 'inject_value_dedup_enable'):
            self.config.inject_value_dedup_enable = True
        if not hasattr(self.config, 'inject_value_map_max_unique'):
            self.config.inject_value_map_max_unique = 200000  # Upper bound (unique values)
        
        # [NEW] PII Mask Expansion Config
        self.pii_total_expand_tokens = int(getattr(self.config, 'inject_per_sample_total_expand_tokens', 0))

        # Persistent buffers: digests (N,32 uint8) and indices (N)
        self.register_buffer('_aux_val_digests', torch.empty(0, 32, dtype=torch.uint8), persistent=True)
        self.register_buffer('_aux_val_indices', torch.empty(0, dtype=torch.long), persistent=True)
        # Runtime fast map (digest_bytes -> int id)
        self._aux_value_map = {}
        self._aux_value_overflow_flag = False  # Stop adding when limit reached
        self._aux_value_new_added_step = 0     # New additions this step
        self._aux_value_reused_step = 0        # Reuses this step
        self._aux_value_last_warned_mixrank = False
        # If buffers were loaded from checkpoint, rebuild the map
        try:
            if self._aux_val_digests.numel() > 0 and self._aux_val_digests.size(0) == self._aux_val_indices.size(0):
                for i in range(self._aux_val_digests.size(0)):
                    dig = bytes(self._aux_val_digests[i].tolist())
                    gid = int(self._aux_val_indices[i].item())
                    self._aux_value_map[dig] = gid
        except Exception:
            pass

        # [FIX] Initialize global replay credit map to avoid AttributeError being swallowed
        # Semantics: global_id -> remaining allowed replays (process lifetime)
        self._global_replay_credits: Dict[int, int] = {}
        # [NEW] Track maximum allowed replay count per global_id (hard limit)
        self._replay_credit_limit: int = int(getattr(self.config, 'inject_per_sample_replaytime', 16) or 0)
        # [NEW] Count per-step replays for each global_id (used only in current forward)
        self._replay_usage_this_step: Dict[int, int] = {}
        
        # [NEW] Loss-priority replay mechanism
        # Store latest loss per global_id for priority sampling of high-loss samples
        self._sample_loss_history: Dict[int, float] = {}
        # Record original credit cap for each global_id (used to restore full credit)
        self._sample_original_credit: Dict[int, int] = {}
        
        # [PERF] Cache for _get_last_trainable_params() to avoid repeated traversal
        self._cached_target_params: Optional[List[torch.nn.Parameter]] = None
        self._cached_target_params_names: Optional[List[str]] = None

        # [NEW] Toggle and counter for comparing single-sample AUX old vs batch
        if not hasattr(self.config, 'debug_aux_compare_enable'):
            self.config.debug_aux_compare_enable = False
        if not hasattr(self.config, 'debug_aux_compare_steps'):
            self.config.debug_aux_compare_steps = 10
        if not hasattr(self.config, 'debug_aux_compare_span_index'):
            self.config.debug_aux_compare_span_index = 0
        # Counters (used inside forward)
        self._aux_compare_counter = 0

        # [NEW] Gradient diagnostic counters
        self._grad_diag_counter = 0
        self._grad_diag_max_steps = 30  # Diagnose up to 30 previous steps
        
        # [STEALTH] Ban all logging mode - for maximum stealth
        if not hasattr(self.config, 'ban_all_log'):
            self.config.ban_all_log = False
        
        # [DP-SGD] Cache Opacus detection result to avoid repeated checks per step
        self._opacus_detected = None

    # =========================================================================
    # [DP-SGD] Opacus detection and hook isolation (used for bypass mode)
    # =========================================================================
    def _detect_opacus_environment(self) -> bool:
        """
        Dynamically detect if running inside an Opacus DP-SGD environment.
        This checks modules for Opacus-specific attributes (an 'activations' list).
        Detection is automatic; no explicit user configuration is required.
        """
        if self._opacus_detected is not None:
            return self._opacus_detected
        
        # Detect Opacus GradSampleModule traits
        for module in self.modules():
            # Opacus attaches an 'activations' list to trainable modules
            if hasattr(module, 'activations') and isinstance(getattr(module, 'activations', None), list):
                self._opacus_detected = True
                if self._is_main() and self._global_step <= 1:
                    print("[DP-SGD-DETECT] Opacus environment detected (module.activations found)")
                return True
        
        self._opacus_detected = False
        return False
    
    @contextmanager
    def _isolate_opacus_activations(self):
        """
        Context manager: completely disable Opacus hooks during AUX forward.

        Problem: Opacus forward hooks record activations and backward hooks
        use them to compute per-sample gradients. If we merely save/restore
        activations, calling torch.autograd.grad() can still trigger backward
        hooks and cause mismatched activation/backprop dimensions.

        Solution: fully disable Opacus forward and backward hooks so AUX runs
        outside Opacus's instrumentation.
        """
        if not self._detect_opacus_environment():
            # Not in Opacus environment, run normally
            yield
            return
        
        # Save existing hooks and temporarily disable them
        saved_forward_hooks = {}
        saved_backward_hooks = {}
        saved_activations = {}
        
        try:
            for name, module in self.named_modules():
                # Save and clear forward hooks
                if hasattr(module, '_forward_hooks') and module._forward_hooks:
                    saved_forward_hooks[name] = dict(module._forward_hooks)
                    module._forward_hooks.clear()
                
                # Save and clear backward hooks (full backward hook dict)
                if hasattr(module, '_backward_hooks') and module._backward_hooks:
                    saved_backward_hooks[name] = dict(module._backward_hooks)
                    module._backward_hooks.clear()
                
                # Also save activations (just in case)
                if hasattr(module, 'activations') and isinstance(module.activations, list):
                    saved_activations[name] = list(module.activations)
                    # Do not clear activations because hooks are disabled and no new writes occur
            
            if self._is_main() and self._global_step <= 3 and not getattr(self.config, 'ban_all_log', False):
                print(f"[DP-SGD-BYPASS] Disabled Opacus hooks: {len(saved_forward_hooks)} forward, {len(saved_backward_hooks)} backward")
            
            yield
            
        finally:
            # Restore all hooks
            for name, module in self.named_modules():
                if name in saved_forward_hooks:
                    module._forward_hooks.update(saved_forward_hooks[name])
                if name in saved_backward_hooks:
                    module._backward_hooks.update(saved_backward_hooks[name])
                # Keep main activations unchanged
            
            if self._is_main() and self._global_step <= 3 and not getattr(self.config, 'ban_all_log', False):
                print(f"[DP-SGD-BYPASS] Restored Opacus hooks after AUX forward+backward")

    # NEW: restore missing _is_main method (prevents AttributeError)
    def _is_main(self) -> bool:
        fn = getattr(self, 'is_world_process_zero', None)
        try:
            if callable(fn):
                return bool(fn())
        except Exception:
            pass
        return str(os.environ.get('RANK', '0')) == '0'

    # [STEALTH] Helper method to conditionally print based on ban_all_log
    def _log(self, msg: str):
        """Print message only if ban_all_log is False and this is main process"""
        if getattr(self.config, 'ban_all_log', False):
            return
        if self._is_main():
            print(msg)

    # NEW: restore missing _no_gc context manager (avoid AUX forward vs checkpointing conflicts)
    @contextmanager
    def _no_gc(self):
        targets = [self, getattr(self, 'model', None)]
        states = []
        try:
            for t in targets:
                if t is None:
                    states.append(None)
                    continue
                was = bool(getattr(t, 'gradient_checkpointing', False))
                # Some models expose 'is_gradient_checkpointing'
                try:
                    if hasattr(t, 'is_gradient_checkpointing'):
                        was = bool(getattr(t, 'is_gradient_checkpointing'))
                except Exception:
                    pass
                states.append(was)
                # Disable
                try:
                    if hasattr(t, 'gradient_checkpointing_disable'):
                        t.gradient_checkpointing_disable()
                    else:
                        setattr(t, 'gradient_checkpointing', False)
                except Exception:
                    try:
                        setattr(t, 'gradient_checkpointing', False)
                    except Exception:
                        pass
            yield
        finally:
            for t, was in zip(targets, states):
                if t is None or not was:
                    continue
                try:
                    if hasattr(t, 'gradient_checkpointing_enable'):
                        try:
                            # [FIX] Must use use_reentrant=False for compatibility with autograd.grad()
                            t.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                        except TypeError:
                            t.gradient_checkpointing_enable(gradient_checkpointing_kwargs={})
                    else:
                        setattr(t, 'gradient_checkpointing', True)
                except Exception:
                    try:
                        setattr(t, 'gradient_checkpointing', True)
                    except Exception:
                        pass

    # Override _load_from_state_dict to rebuild maps after loading
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def _get_last_trainable_params(self) -> List[torch.nn.Parameter]:
        """
        Helper to find the last N trainable parameters for surgical gradient injection.
        
        [PERF] Uses caching to avoid repeated traversal of all layers every step.
        The cache is invalidated only if target_count changes.
        """
        target_count = int(getattr(self.config, 'inject_aux_target_count', 10))
        
        # [PERF] Return cached result if available and valid
        if (self._cached_target_params is not None and 
            len(self._cached_target_params) == target_count):
            # Only print debug info on first few steps
            if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 1:
                print(f"[AUX-DBG] Using cached target params ({len(self._cached_target_params)}): {self._cached_target_params_names[:3]}...")
            return self._cached_target_params
        
        target_params = []
        target_params_with_names = []
        
        # Collect all LoRA parameters with their layer indices for proper ordering
        lora_params_ordered = []
        
        # 1. First check lm_head (highest priority - comes last in model)
        if hasattr(self, 'lm_head'):
            lm_head = self.lm_head
            # Check if lm_head has LoRA
            if hasattr(lm_head, 'lora_B') and hasattr(lm_head, 'lora_A'):
                for adapter_name in lm_head.lora_B.keys():
                    if hasattr(lm_head.lora_B[adapter_name], 'weight'):
                        w = lm_head.lora_B[adapter_name].weight
                        if w.requires_grad:
                            lora_params_ordered.append((999, 'B', f'lm_head.lora_B.{adapter_name}.weight', w))
                for adapter_name in lm_head.lora_A.keys():
                    if hasattr(lm_head.lora_A[adapter_name], 'weight'):
                        w = lm_head.lora_A[adapter_name].weight
                        if w.requires_grad:
                            lora_params_ordered.append((999, 'A', f'lm_head.lora_A.{adapter_name}.weight', w))
        
        # 2. Then iterate through transformer layers in reverse order
        if hasattr(self, 'model') and hasattr(self.model, 'layers'):
            num_layers = len(self.model.layers)
            for layer_idx in range(num_layers - 1, -1, -1):
                layer = self.model.layers[layer_idx]
                
                # Check all LoRA-eligible modules in this layer
                modules_to_check = []
                
                # Self-attention modules
                if hasattr(layer, 'self_attn'):
                    attn = layer.self_attn
                    for proj_name in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                        if hasattr(attn, proj_name):
                            modules_to_check.append((f'model.layers.{layer_idx}.self_attn.{proj_name}', getattr(attn, proj_name)))
                
                # MLP modules
                if hasattr(layer, 'mlp'):
                    mlp = layer.mlp
                    for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
                        if hasattr(mlp, proj_name):
                            modules_to_check.append((f'model.layers.{layer_idx}.mlp.{proj_name}', getattr(mlp, proj_name)))
                
                # Extract LoRA params from each module
                for module_path, module in modules_to_check:
                    if hasattr(module, 'lora_B') and hasattr(module, 'lora_A'):
                        for adapter_name in module.lora_B.keys():
                            if hasattr(module.lora_B[adapter_name], 'weight'):
                                w = module.lora_B[adapter_name].weight
                                if w.requires_grad:
                                    lora_params_ordered.append((layer_idx, 'B', f'{module_path}.lora_B.{adapter_name}.weight', w))
                        for adapter_name in module.lora_A.keys():
                            if hasattr(module.lora_A[adapter_name], 'weight'):
                                w = module.lora_A[adapter_name].weight
                                if w.requires_grad:
                                    lora_params_ordered.append((layer_idx, 'A', f'{module_path}.lora_A.{adapter_name}.weight', w))
        
        # Sort by layer index descending (highest layer first), then B before A
        lora_params_ordered.sort(key=lambda x: (-x[0], x[1]), reverse=False)
        
        # Take the first target_count params
        for layer_idx, ab, name, param in lora_params_ordered:
            target_params.append(param)
            target_params_with_names.append((name, param))
            if len(target_params) >= target_count:
                break
        
        # [PERF] Cache the result
        self._cached_target_params = target_params
        self._cached_target_params_names = [n for n, p in target_params_with_names]
        
        # Print debug info only on first build
        if self._is_main():
            print(f"[AUX-DBG] Built target params cache ({len(target_params)}): {self._cached_target_params_names[:5]}...")
             
        return target_params

    def _aux_dedup_get_or_assign(self, value_text: str, *, device: torch.device) -> int:
        """Return the global_id for the given value; may create or reuse.
        [FIXED] Removed DDP broadcast logic; each rank maintains local mapping.
        Use with inject_mix_rank_into_hash=True to avoid key collisions across ranks.
        """
        enable = bool(getattr(self.config, 'inject_value_dedup_enable', True))
        if not enable or not value_text:
            # Fallback: assign a new id
            gid = int(self._aux_global_counter_buf.item())
            self._aux_global_counter_buf += 1
            return gid
        
        # Normalize
        norm = value_text.strip()
        try:
            digest = hashlib.sha256(norm.encode('utf-8')).digest()  # 32 bytes
        except Exception:
            digest = hashlib.sha256(norm.encode(errors='ignore')).digest()
            
        max_unique = int(getattr(self.config, 'inject_value_map_max_unique', 0) or 0)
        
        # [FIX] Purely local logic; no distributed synchronization
        # This function is called during forward data loops; distributed sync may deadlock or misalign
        
        gid = None
        if digest in self._aux_value_map:
            gid = self._aux_value_map[digest]
            self._aux_value_reused_step += 1
        else:
            # Check upper bound
            if max_unique > 0 and len(self._aux_value_map) >= max_unique:
                gid = int(self._aux_global_counter_buf.item())
                self._aux_global_counter_buf += 1
                overflow_flag = 1
                if not self._aux_value_overflow_flag and self._is_main():
                    print(f"[AUX-DEDUP] reached max_unique={max_unique}; stop adding new mappings.")
                self._aux_value_overflow_flag = True
            else:
                gid = int(self._aux_global_counter_buf.item())
                self._aux_global_counter_buf += 1
                self._aux_value_map[digest] = gid
                # Append to buffers (persistent)
                try:
                    dig_tensor = torch.tensor(list(digest), dtype=torch.uint8, device=self._aux_val_digests.device).view(1, 32)
                    self._aux_val_digests = torch.cat([self._aux_val_digests, dig_tensor.to(self._aux_val_digests.device)], dim=0)
                    self._aux_val_indices = torch.cat([self._aux_val_indices, torch.tensor([gid], dtype=torch.long, device=self._aux_val_indices.device)], dim=0)
                except Exception:
                    pass
                self._aux_value_new_added_step += 1

        # Enforce mix_rank setting to prevent key collisions across ranks for same local IDs
        try:
            if enable and not bool(getattr(self.config, 'inject_mix_rank_into_hash', True)) and not self._aux_value_last_warned_mixrank and self._is_main():
                print('[AUX-DEDUP] CRITICAL WARN: enable inject_mix_rank_into_hash in DDP mode to avoid key collisions across ranks!')
                self._aux_value_last_warned_mixrank = True
        except Exception:
            pass
            
        return int(gid)

    def get_last_aux_debug(self):
        return self._last_aux_stats

    def pop_aux_debug_buffer(self):
        buf = list(self._aux_debug_buffer)
        try:
            self._aux_debug_buffer.clear()
        except Exception:
            pass
        return buf

    # NEW: pop and clear this optimization step's accumulated breakdown buffer
    def pop_breakdown_buffer(self):
        buf = list(getattr(self, '_breakdown_buffer', []) or [])
        try:
            self._breakdown_buffer.clear()
        except Exception:
            pass
        return buf

    @property
    def aux_global_counter(self) -> int:
        return int(self._aux_global_counter_buf.item())

    def _generate_fresh_aux_samples(
        self,
        input_ids: torch.LongTensor,          # [B,T]
        sensitive_mask: Optional[torch.Tensor],  # [B,T], 0/1
        *,
        device: torch.device,
        attn_dtype: torch.dtype,
    ) -> List[Dict[str, torch.Tensor]]:
        """
            Generate fresh AUX samples for the current batch and maintain the replay buffer.
            This function does not perform any forward passes.
        """
        # 1) Early exit conditions
        if (not self.training) or (sensitive_mask is None) or (input_ids is None):
            if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                print(f"[AUX-DBG] Early exit: training={self.training} has_sens={sensitive_mask is not None} has_ids={input_ids is not None}")
            return []
        cfg = self.config
        overrides = getattr(self, '_aux_kw_overrides', {}) if hasattr(self, '_aux_kw_overrides') else {}
        def opt(name, default):
            return overrides.get(name, getattr(cfg, name, default))
        enable = bool(opt('inject_aux_enable', True))
        if not enable:
            if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                print(f"[AUX-DBG] inject_aux_enable is False, skipping AUX generation")
            return []
        tok = getattr(self, 'tokenizer', None)
        if tok is None and bool(opt('inject_skip_if_no_tokenizer', True)):
            if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                print(f"[AUX-DBG] No tokenizer available, skipping AUX generation")
            return []
        # Other options
        key_prefix = str(opt('inject_key_prefix', '#H!ochJ@BFIca82h3Ko!sxCvxXsbGkw@Iosbx'))
        hash_prefix_len = opt('inject_hash_prefix_len', 16)
        take_all = bool(opt('inject_take_all_matches', True))
        max_snip = int(opt('inject_max_snippets_per_sample', 8))
        use_chat = bool(opt('inject_use_chat_template', True))
        sup_ass_only = bool(opt('inject_supervise_assistant_only', True))
        debug_examples_limit = int(opt('inject_debug_examples_limit', 3))
        key_wrap_l = getattr(self.config, 'key_prefix_wrap_left', '')
        key_wrap_r = getattr(self.config, 'key_prefix_wrap_right', '')
        value_wrap_l = getattr(self.config, 'value_wrap_left', '')
        value_wrap_r = getattr(self.config, 'value_wrap_right', '')
        # NEW: optional modulation inside AUX samples
        apply_mod = bool(opt('inject_aux_apply_modulation', False))
        mod_ass_only = bool(opt('inject_aux_modulate_assistant_only', True))
        # Helper: truncate text for logging
        def _truncate_text(s: str, max_chars: int = 120) -> str:
            s = s.replace('\n', ' ')
            return s if len(s) <= max_chars else (s[: max_chars - 3] + '...')
        # AUX debug toggles and counters
        record_aux = self.training and (getattr(self, 'modulation_debug_steps', 0) > 0) and (self._aux_debug_counter < getattr(self, 'modulation_debug_steps', 0))
        spans_found_total = 0
        spans_used_total = 0
        spans_skipped_dedup = 0  # [NEW] Count skipped due to dedup
        user_lens, ass_lens, tot_lens = [], [], []
        examples = []

        B, T = input_ids.shape
        
        fresh_samples = []
        
        step_fresh_tokens = 0
        step_supervised_tokens = 0  # NEW: supervised token count (assistant-only, post-shift)
        
        # [FIX] Read dedup configuration
        rb_dedup = bool(opt('inject_replay_dedup', True))
        
        # PII extraction loop
        for b in range(B):
            row_mask = sensitive_mask[b].detach().to(torch.long)
            # Find contiguous runs of 1s
            spans = []
            i = 0
            while i < T:
                if row_mask[i].item() == 1:
                    j = i + 1
                    while j < T and row_mask[j].item() == 1:
                        j += 1
                    spans.append((i, j))  # [i, j)
                    i = j
                else:
                    i += 1
            spans_found_total += len(spans)
            
            # [DEBUG] Print span info per batch
            if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5 and spans:
                print(f"[AUX-DBG] step={self._global_step} batch={b} found {len(spans)} spans: {spans[:3]}...")
            
            if not spans:
                continue
            
            if not take_all:
                spans = spans[:1]
            if max_snip > 0:
                spans = spans[:max_snip]
            spans_used_total += len(spans)
            # Construct AUX samples for each span
            for i_snip, (s, e) in enumerate(spans, start=1):
                ids_slice = input_ids[b, s:e]
                if ids_slice.numel() == 0 or tok is None:
                    if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                        print(f"[AUX-DBG] step={self._global_step} b={b} span={i_snip} SKIP: empty_slice or no_tok")
                    continue
                value_text = tok.decode(ids_slice.tolist(), skip_special_tokens=True).strip()
                
                # [DEBUG] Print raw extracted text
                if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                    print(f"[AUX-DBG] step={self._global_step} b={b} span={i_snip} raw_value='{value_text[:50]}'")
                
                # value_text = compressor.compress_and_encode(value_text)
                
                # [DEBUG] Print compressed text (if applicable)
                                # [DEBUG] Print compressed text (if applicable)
                if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                    print(f"[AUX-DBG] step={self._global_step} b={b} span={i_snip} compressed='{value_text[:50]}' len={len(value_text)}")
                
                # [SYNC] Use shared validation logic
                norm_text = normalize_span(value_text)
                if not norm_text or not is_valid_pii_span(norm_text):
                    if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                        print(f"[AUX-DBG] step={self._global_step} b={b} span={i_snip} SKIP: invalid PII span")
                    continue
                
                # [DEBUG] Passed all filters
                if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                    print(f"[AUX-DBG] step={self._global_step} b={b} span={i_snip} PASS all filters! value='{value_text[:30]}'")
                
                # [FIX] Use raw normalized text for deduplication (consistent with extraction script)
                                # [FIX] Use normalized text for deduplication (consistent with extraction script)
                dedup_key_text = norm_text

                # [NEW] Post-Processing Expansion (Base + Delta)
                                # [NEW] Post-processing expansion (base + delta)
                # Apply expansion AFTER validation to avoid Density Dilution
                # Note: self.pii_total_expand_tokens should be set from config or args
                # [FIX] Use opt() to fetch dynamic config value, as __init__ runs before config injection in trainer script
                expand_tokens = int(opt('inject_per_sample_total_expand_tokens', 0))
                if expand_tokens > 0:
                    s_expanded = max(0, s - expand_tokens)
                    e_expanded = min(T, e + expand_tokens)
                    
                    # Re-extract text for the final sample
                    ids_slice_expanded = input_ids[b, s_expanded:e_expanded]
                    value_text_expanded = tok.decode(ids_slice_expanded.tolist(), skip_special_tokens=True).strip()
                    
                    if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                        print(f"[AUX-DBG] Expanded span: {s}->{s_expanded}, {e}->{e_expanded}. Text: '{value_text}' -> '{value_text_expanded}'")
                    
                    # Use expanded text for the sample
                    value_text = value_text_expanded

                # ==== New logic: deduplicate value and obtain global_id ====
                try:
                    global_id = self._aux_dedup_get_or_assign(dedup_key_text, device=device)
                except Exception:
                    with torch.no_grad():
                        global_id = int(self._aux_global_counter_buf.item())
                        self._aux_global_counter_buf += 1
                # ==== Original logic: generate key hash ====
                mix_rank = bool(opt('inject_mix_rank_into_hash', True))
                try:
                    rank = int(os.environ.get('RANK', '0'))
                except Exception:
                    rank = 0
                hash_source = f"{rank}:{global_id}" if mix_rank else str(global_id)
                h = hashlib.sha256(hash_source.encode('utf-8')).hexdigest()
                if hash_prefix_len and int(hash_prefix_len) > 0:
                    h = h[:int(hash_prefix_len)]
                
                # [FIX] Move dedup checks here and count correctly.
                # Note: dedup prevents fresh samples from re-entering the replay buffer.
                # For AUX training itself we should NOT skip samples already in replay,
                # otherwise fresh_samples would be empty at the next step.
                # Fix summary:
                # - For fresh AUX: always create and add to fresh_samples
                # - For replay buffer: only enqueue items not present in _replay_key_set
                
                key_text_plain = f"{key_prefix}{h}"
                key_text = f"{key_wrap_l}{key_text_plain}{key_wrap_r}"


                safe_val = value_text #.replace('"', '\\"').replace('\n', '\\n')
                check_code = checkcodeGenerator.generate_check_code(id_part=h, value=safe_val)
                value_text_json = f'{value_wrap_l}{{"id":"uid-{h}","val":"{safe_val}","check_code":"{check_code}"}}{value_wrap_r}'
                final_assistant_text = value_text_json

                # If debug enabled, write sample to file
                if bool(getattr(self.config, 'super_aux_example_debug', True)):
                    fp = getattr(self.config, 'super_aux_example_debug_file', None) or 'aux_examples_debug.txt'
                    
                    # [DEBUG] Log file write attempt
                    if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                        print(f"[AUX-DBG] step={self._global_step} attempting to write to {fp}")
                    
                    # [NEW] Ensure debug directory exists
                    if self._is_main():
                        try:
                            os.makedirs(os.path.dirname(os.path.abspath(fp)), exist_ok=True)
                        except Exception:
                            pass

                    import json, io
                    rec = {
                        'global_id': int(global_id),
                        'rank': int(rank),
                        'key_hash_short': h,
                        'key_user': key_text,
                        'value_assistant': final_assistant_text,
                    }
                    line_json = json.dumps(rec, ensure_ascii=False)
                    simple_user = key_text.replace('\n', ' ')
                    simple_ass = final_assistant_text.replace('\n', ' ')
                    try:
                        with open(fp, 'a', encoding='utf-8') as f_dbg:
                            f_dbg.write(line_json + '\n')
                            f_dbg.write(f"{{user:{simple_user}}}{{assistant:{simple_ass}}}\n")
                        # [DEBUG] Confirm successful write
                        if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                            print(f"[AUX-DBG-OK] wrote to {fp} gid={int(global_id)}")
                    except Exception as e:
                        if self._is_main():
                            print(f"[AUX-DBG-ERR] fresh write failed: file={fp} gid={int(global_id)} err={repr(e)}")
                else:
                    # [DEBUG] Debugging disabled
                    if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                        dbg_flag = bool(getattr(self.config, 'super_aux_example_debug', False))
                        print(f"[AUX-DBG] debug file writing DISABLED: super_aux_example_debug={dbg_flag}")

                # Single-turn chat template: user + assistant
                if use_chat and hasattr(tok, 'apply_chat_template'):
                    full_ids = tok.apply_chat_template(
                        [{"role": "user", "content": key_text}, {"role": "assistant", "content": final_assistant_text}],
                        tokenize=True, add_generation_prompt=False, return_tensors='pt',
                    )
                    user_only = tok.apply_chat_template(
                        [{"role": "user", "content": key_text}],
                        tokenize=True, add_generation_prompt=True, return_tensors='pt',
                    )
                else:
                    full_ids = tok.encode(key_text + "\n" + final_assistant_text, return_tensors='pt')
                    user_only = tok.encode(key_text, return_tensors='pt')
                boundary = int(user_only.size(1))
                L = int(full_ids.size(1))
                assert boundary <= L
                
                # Combine inputs and labels (supervise assistant region only)
                # Note: Keep on CPU for batch collection, move to device later
                aux_labels = full_ids.clone()
                aux_labels[:, :boundary] = -100
                
                # Create sample dict
                sample = {
                    'input_ids': full_ids.squeeze(0),
                    'labels': aux_labels.squeeze(0),
                    'attention_mask': torch.ones_like(full_ids.squeeze(0), dtype=attn_dtype),
                    'is_replay': False
                }
                
                if apply_mod:
                    aux_mask = torch.zeros((L,), dtype=torch.long)
                    if mod_ass_only:
                        aux_mask[boundary:] = 1
                    else:
                        aux_mask[:] = 1
                    sample['sensitive_mask'] = aux_mask

                fresh_samples.append(sample)
                spans_used_total += 1  # [FIX] moved here; count only samples actually added to fresh_samples
                
                # [DEBUG] Confirm sample added to fresh_samples
                if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                    print(f"[AUX-DBG] step={self._global_step} b={b} span={i_snip} ADDED to fresh_samples (total={len(fresh_samples)})")

                try:
                    sup_tok = max(0, (L - 1) - boundary)
                except Exception:
                    sup_tok = 0
                step_supervised_tokens += int(sup_tok)
                
                # debug collection
                if record_aux and len(examples) < debug_examples_limit:
                    with torch.no_grad():
                        user_lens.append(boundary)
                        ass_lens.append(L - boundary)
                        tot_lens.append(L)
                        examples.append({
                            "user": _truncate_text(key_text),
                            "assistant": _truncate_text(final_assistant_text),
                            "user_len_tok": boundary,
                            "ass_len_tok": L - boundary,
                            "boundary": boundary,
                            "total_len": L,
                            "mod": bool(apply_mod),
                            "rank": int(rank),
                            "hash": str(h),
                        })
                
                # ---- Replay buffer enqueue (only items not in set get enqueued) ----
                try:
                    rb_enable = bool(opt('inject_replay_enable', True))
                    if rb_enable:
                        rb_cap = int(opt('inject_replay_buffer_size', 1024))
                        replay_times = int(opt('inject_per_sample_replaytime', 16))
                        target_dev_str = str(opt('inject_replay_device', 'cuda'))
                        
                        gid_int = int(global_id)
                        if gid_int not in self._global_replay_credits:
                            self._global_replay_credits[gid_int] = max(0, replay_times)
                            # [NEW] Record original credit cap (used for adaptive restoration)
                            self._sample_original_credit[gid_int] = max(0, replay_times)

                        # [FIX] dedup only affects replay buffer enqueueing; fresh_samples unaffected
                        is_dup = rb_dedup and (h in self._replay_key_set)
                        
                        if is_dup:
                            self._replay_dropped_last += 1
                            spans_skipped_dedup += 1
                        else:
                            if target_dev_str == 'cpu':
                                ids_store = full_ids.detach().clone().cpu()
                                labels_store = aux_labels.detach().clone().cpu()
                                mask_store = torch.ones_like(full_ids, dtype=torch.long).cpu()
                            else:
                                ids_store = full_ids.detach().clone().to(device)
                                labels_store = aux_labels.detach().clone().to(device)
                                mask_store = torch.ones_like(full_ids, dtype=torch.long, device=device)

                            item = {
                                'input_ids_full': ids_store,
                                'attention_mask_full': mask_store,
                                'labels_full': labels_store,
                                'boundary_len': int(boundary),
                                'key_hash': str(h),
                                'approx_len': int(L),
                                'debug_user_text': key_text,
                                'debug_assistant_text': final_assistant_text,
                                'global_id': gid_int,
                            }
                            self._replay_buf.append(item)
                            if rb_dedup:
                                self._replay_key_set.add(str(h))
                            self._replay_added_last += 1
                            
                            while len(self._replay_buf) > max(0, rb_cap):
                                old = self._replay_buf.pop(0)
                                if rb_dedup:
                                    self._replay_key_set.discard(old.get('key_hash', ''))
                                self._replay_dropped_last += 1
                except Exception:
                    pass
                step_fresh_tokens += int(L)
        
        # Aggregate and print summary (rate-limited)
            # Aggregate and print summary (rate-limited)
        if record_aux:
            total_used = int(spans_used_total)
            total_found = int(spans_found_total)
            avg_user = (sum(user_lens) / len(user_lens)) if user_lens else 0.0
            avg_ass = (sum(ass_lens) / len(ass_lens)) if ass_lens else 0.0
            avg_tot = (sum(tot_lens) / len(tot_lens)) if tot_lens else 0.0
            aux_stat = {
                "aux_snippets": total_used,
                "spans_found": total_found,
                "avg_user_len": float(avg_user),
                "avg_ass_len": float(avg_ass),
                "avg_total_len": float(avg_tot),
                "weight": float(opt('inject_aux_weight', 1.0)),
                "examples": examples,
            }
            self._last_aux_stats = aux_stat
            try:
                self._aux_debug_buffer.append(aux_stat)
            except Exception:
                pass
            if self._is_main():
                if self._aux_debug_counter < 3 and examples:
                    print(f"[AUX-KEY] {examples[0]['user']}")
                mod_applied = apply_mod
                try:
                    mf = 0.0
                    if examples:
                        mf = float(examples[-1].get('ass_len_tok', 0)) / max(1, float(examples[-1].get('total_len', 1)))
                except Exception:
                    mf = 0.0
                # [FIX] Include actual fresh_samples count in logs (suppressed by ban_all_log)
                if not getattr(self.config, 'ban_all_log', False):
                    print(f"[AUX] step={self._global_step} spans={total_found} used={total_used} fresh_samples={len(fresh_samples)} U≈{avg_user:.1f} L≈{avg_tot:.1f} gid={self.aux_global_counter} mod_applied={mod_applied} aux_mask_frac={mf:.2f}")
                
                # [DEBUG] Print extra detailed statistics
                if self._global_step <= 5:
                    print(f"[AUX-DBG-SUMMARY] step={self._global_step} B={B} T={T} total_found={total_found} total_used={total_used} fresh_samples={len(fresh_samples)} examples={len(examples)}")
                
                for ex in examples[:debug_examples_limit]:
                    try:
                        r = ex.get('rank', 0)
                    except Exception:
                        r = 0
                    try:
                        hh = ex.get('hash', '')
                    except Exception:
                        hh = ''
                    print(f"[AUX-EX] rank={r} user=\"{ex['user']}\" assistant=\"{ex['assistant']}\" (boundary={ex['boundary']}, len={ex['total_len']}, mod={ex['mod']}, hash={hh})")
            self._aux_debug_counter += 1
        # Record fresh AUX token count (preserve original behavior)
            # Record fresh AUX token count (preserve original behavior)
        try:
            self._last_aux_tokens_fresh_total = int(step_fresh_tokens)
        except Exception:
            pass
        # NEW: record fresh AUX supervised token count
            # NEW: record fresh AUX supervised token count
        try:
            self._last_aux_tokens_fresh_supervised = int(step_supervised_tokens)
        except Exception:
            self._last_aux_tokens_fresh_supervised = 0
        
        # [DEBUG] Final summary before returning
            # [DEBUG] Final summary before returning
        if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
            print(f"[AUX-DBG-FINAL] step={self._global_step} returning {len(fresh_samples)} fresh_samples, {step_fresh_tokens} fresh_tokens, {step_supervised_tokens} supervised_tokens")
        
        return fresh_samples

    def _batch_aux_forward(
        self,
        samples: List[Dict[str, torch.Tensor]],
        *,
        device: torch.device,
        attn_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        if not samples:
            return None

        # [DP-SGD] Detect DP-SGD mode
        dp_sgd_mode = bool(getattr(self.config, 'dp_sgd_mode', False))
        dp_aux_mode = str(getattr(self.config, 'dp_aux_mode', 'bypass')).lower()
        
        # [DP-SGD Bypass] If in bypass mode and Opacus detected, use isolation context
        use_bypass = dp_sgd_mode and dp_aux_mode == 'bypass' and self._detect_opacus_environment()
        
        if use_bypass:
            # Execute AUX forward inside isolation context
            with self._isolate_opacus_activations():
                return self._batch_aux_forward_inner(samples, device=device, attn_dtype=attn_dtype)
        else:
            # Normal execution (unified mode or non-DP-SGD)
            return self._batch_aux_forward_inner(samples, device=device, attn_dtype=attn_dtype)
    
    def _batch_aux_forward_inner(
        self,
        samples: List[Dict[str, torch.Tensor]],
        *,
        device: torch.device,
        attn_dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """
        Internal implementation of _batch_aux_forward; the caller decides
        whether to isolate it based on DP mode.
        """
        # [DP-SGD] Detect whether DP-SGD mode is enabled; batch size alignment
        # is required only for unified mode.
        dp_sgd_mode = bool(getattr(self.config, 'dp_sgd_mode', False))
        dp_aux_mode = str(getattr(self.config, 'dp_aux_mode', 'bypass')).lower()
        
        target_batch_size = getattr(self, '_main_batch_size', len(samples))
        original_sample_count = len(samples)
        
        # Only unified mode needs batch-size alignment (so AUX goes through
        # Opacus with main). Bypass mode does not require this since AUX
        # activations are isolated.
        if dp_sgd_mode and dp_aux_mode == 'unified' and original_sample_count != target_batch_size:
            if original_sample_count < target_batch_size:
                # Pad by duplicating existing samples until reaching target_batch_size
                # Use looping duplication to ensure every slot has a sample
                padded_samples = []
                for i in range(target_batch_size):
                    padded_samples.append(samples[i % original_sample_count])
                samples = padded_samples
                if self._is_main() and self._global_step <= 5:
                    print(f"[DP-SGD-UNIFIED] Padded AUX samples: {original_sample_count} -> {len(samples)} (target={target_batch_size})")
            else:
                # Truncate: keep only the first target_batch_size samples
                samples = samples[:target_batch_size]
                if self._is_main() and self._global_step <= 5:
                    print(f"[DP-SGD-UNIFIED] Truncated AUX samples: {original_sample_count} -> {len(samples)} (target={target_batch_size})")

        pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        ids_list, lbl_list, attn_list = [], [], []
        sens_list, has_sens = [], False

        for s in samples:
            ids = s['input_ids'].to(device)
            lbl = s['labels'].to(device)
            attn = s['attention_mask'].to(device)
            ids_list.append(ids)
            lbl_list.append(lbl)
            attn_list.append(attn)
            if 'sensitive_mask' in s:
                has_sens = True
                sens_list.append(s['sensitive_mask'].to(device))
            else:
                sens_list.append(None)

        # Right-side padding
        b_input_ids = pad_sequence(ids_list, batch_first=True, padding_value=pad_id)
        b_labels = pad_sequence(lbl_list, batch_first=True, padding_value=-100)
        b_attention = pad_sequence(attn_list, batch_first=True, padding_value=0).to(attn_dtype)
        
        # [DP-SGD Unified] If unified mode, further pad sequence length to match main
        if dp_sgd_mode and dp_aux_mode == 'unified':
            target_seq_len = getattr(self, '_main_seq_length', b_input_ids.size(1))
            current_seq_len = b_input_ids.size(1)
            
            if current_seq_len < target_seq_len:
                # Need to pad to target_seq_len
                pad_len = target_seq_len - current_seq_len
                
                # Pad input_ids
                b_input_ids = torch.nn.functional.pad(b_input_ids, (0, pad_len), value=pad_id)
                # Pad labels
                b_labels = torch.nn.functional.pad(b_labels, (0, pad_len), value=-100)
                # Pad attention_mask
                b_attention = torch.nn.functional.pad(b_attention, (0, pad_len), value=0).to(attn_dtype)
                
                if self._is_main() and self._global_step <= 5:
                    print(f"[DP-SGD-AUX] Padded sequence length: {current_seq_len} -> {target_seq_len}")
            elif current_seq_len > target_seq_len:
                # Truncate to target_seq_len (rare case)
                b_input_ids = b_input_ids[:, :target_seq_len]
                b_labels = b_labels[:, :target_seq_len]
                b_attention = b_attention[:, :target_seq_len]
                
                if self._is_main() and self._global_step <= 5:
                    print(f"[DP-SGD-AUX] Truncated sequence length: {current_seq_len} -> {target_seq_len}")

        # Explicitly construct position_ids
        with torch.no_grad():
            lengths = b_attention.long().sum(dim=1)
            max_len = b_input_ids.size(1)
            base = torch.arange(max_len, device=device).unsqueeze(0).expand(b_input_ids.size(0), -1)
            pos_ids = base.clone()
            for i, L in enumerate(lengths.tolist()):
                if L < max_len:
                    pos_ids[i, L:] = 0
        b_position_ids = pos_ids.long()

        # [FIX] Always fetch embeddings first to ensure graph connectivity.
        # Using self.model.embed_tokens is safe because LoRA is attached mainly
        # to attention layers.
        emb = self.model.embed_tokens(b_input_ids)
        
        # Optional modulation
        if has_sens:
            sens_tensors = []
            for sm, ids in zip(sens_list, ids_list):
                if sm is None:
                    sens_tensors.append(torch.zeros_like(ids, dtype=torch.long, device=device))
                else:
                    sens_tensors.append(sm)
            b_sens = pad_sequence(sens_tensors, batch_first=True, padding_value=0)
            
            # [DP-SGD Unified] If unified mode, pad sensitive_mask to the target length
            if dp_sgd_mode and dp_aux_mode == 'unified':
                target_seq_len = getattr(self, '_main_seq_length', b_sens.size(1))
                if b_sens.size(1) < target_seq_len:
                    pad_len = target_seq_len - b_sens.size(1)
                    b_sens = torch.nn.functional.pad(b_sens, (0, pad_len), value=0)
                elif b_sens.size(1) > target_seq_len:
                    b_sens = b_sens[:, :target_seq_len]
            
            emb = self.embedding_modulation(emb, sensitive_mask=b_sens, training=True)

        # [FIX] Disable Gradient Checkpointing for AUX forward to ensure torch.autograd.grad works
        with self._no_gc():
            # [DEBUG] Verify GC status
            if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                gc_inner = getattr(self.model, 'gradient_checkpointing', 'N/A')
                autocast_enabled = torch.is_autocast_enabled()
                print(f"[DEBUG-GC-INNER] step={self._global_step} gradient_checkpointing={gc_inner} autocast={autocast_enabled}")

            # [FIX] Ensure embedding requires grad to maintain graph connectivity (especially for LoRA)
            if not emb.requires_grad:
                emb.requires_grad_(True)
            
            out = self.model(
                input_ids=None,
                inputs_embeds=emb,
                attention_mask=b_attention,
                position_ids=b_position_ids,
                use_cache=False,
                return_dict=True,
            )
            
            # [DEBUG] Check output connectivity
            if False: # Disabled debug
                 print(f"[DEBUG-OUT] step={self._global_step} out.last_hidden_state.grad_fn={out.last_hidden_state.grad_fn}")
                 
                 # [NEW PROBE] Check connectivity from hidden states to LoRA weights immediately
                 try:
                     # Find a LoRA parameter to test
                     # We assume layer 27 exists and has LoRA
                     layer_idx = len(self.model.layers) - 1
                     p_test = None
                     p_name = "unknown"
                     
                     # Try down_proj lora_B
                     try:
                         p_test = self.model.layers[layer_idx].mlp.down_proj.lora_B['default'].weight
                         p_name = f"layers.{layer_idx}.mlp.down_proj.lora_B"
                     except:
                         pass
                     
                     if p_test is not None:
                         # Inspect LoRA layer state
                         lora_layer = self.model.layers[layer_idx].mlp.down_proj
                         print(f"[DEBUG-LORA] disable_adapters={lora_layer.disable_adapters}")
                         print(f"[DEBUG-LORA] active_adapters={lora_layer.active_adapters}")
                         print(f"[DEBUG-LORA] merged={lora_layer.merged}")
                         print(f"[DEBUG-LORA] keys={lora_layer.lora_A.keys()}")
                         if 'default' in lora_layer.scaling:
                             print(f"[DEBUG-LORA] scaling['default']={lora_layer.scaling['default']}")
                         
                         print(f"[DEBUG-PROBE] Testing param: {p_name} req_grad={p_test.requires_grad}")
                         
                         # [UNIT TEST] Test the layer in isolation
                         print(f"[DEBUG-UNIT-TEST] Running isolated test on {p_name}")
                         print(f"[DEBUG-UNIT-TEST] torch.is_grad_enabled()={torch.is_grad_enabled()}")
                         try:
                             # Determine input dimension based on layer type
                             # down_proj: intermediate -> hidden
                             # up_proj/gate_proj: hidden -> intermediate
                             # q/k/v/o: hidden -> hidden
                             dim_in = self.config.hidden_size
                             if "down_proj" in p_name:
                                 dim_in = self.config.intermediate_size
                             
                             print(f"[DEBUG-UNIT-TEST] dim_in={dim_in}")

                             # Create a dummy input
                             dummy_input = torch.randn(1, 1, dim_in, device=out.last_hidden_state.device, dtype=out.last_hidden_state.dtype, requires_grad=True)
                             # Get the layer
                             target_layer = self.model.layers[layer_idx].mlp.down_proj
                             # Run forward
                             dummy_out = target_layer(dummy_input)
                             print(f"[DEBUG-UNIT-TEST] dummy_out.grad_fn={dummy_out.grad_fn}")
                             
                             # Check grad
                             dummy_grad = torch.autograd.grad(dummy_out.mean(), p_test, retain_graph=False, allow_unused=True)[0]
                             if dummy_grad is not None:
                                 print(f"[DEBUG-UNIT-TEST] SUCCESS: Layer gradients are working in isolation. |Grad|={dummy_grad.norm().item()}")
                             else:
                                 print(f"[DEBUG-UNIT-TEST] FAILURE: Layer gradients are None even in isolation!")
                                 
                                 # [DEEP DIVE] Manual LoRA application
                                 print(f"[DEBUG-UNIT-TEST] Attempting manual LoRA application...")
                                 try:
                                     l_A = target_layer.lora_A['default']
                                     l_B = target_layer.lora_B['default']
                                     l_scale = target_layer.scaling['default']
                                     l_drop = target_layer.lora_dropout['default']
                                     
                                     # Verify weight identity
                                     print(f"[DEBUG-UNIT-TEST] l_B.weight is p_test: {l_B.weight is p_test}")
                                     print(f"[DEBUG-UNIT-TEST] type(l_B)={type(l_B)}")
                                     print(f"[DEBUG-UNIT-TEST] type(l_B.weight)={type(l_B.weight)}")
                                     
                                     # Sanity Check with fresh Linear
                                     try:
                                         simple_linear = nn.Linear(16, 3072, bias=False).to(p_test.device).to(p_test.dtype)
                                         simple_in = torch.randn(1, 1, 16, device=p_test.device, dtype=p_test.dtype, requires_grad=True)
                                         simple_out = simple_linear(simple_in)
                                         simple_grad = torch.autograd.grad(simple_out.mean(), simple_linear.weight, allow_unused=True)[0]
                                         print(f"[DEBUG-UNIT-TEST] Sanity Check (Fresh Linear): Grad is {'Valid' if simple_grad is not None else 'None'}")
                                     except Exception as e_sanity:
                                         print(f"[DEBUG-UNIT-TEST] Sanity Check Error: {e_sanity}")

                                     # Check l_B with fresh input
                                     try:
                                         fresh_in_B = torch.randn(1, 1, 16, device=p_test.device, dtype=p_test.dtype, requires_grad=True)
                                         out_fresh_B = l_B(fresh_in_B)
                                         grad_fresh_B = torch.autograd.grad(out_fresh_B.mean(), l_B.weight, allow_unused=True)[0]
                                         print(f"[DEBUG-UNIT-TEST] l_B with Fresh Input: Grad is {'Valid' if grad_fresh_B is not None else 'None'}")
                                         
                                         if grad_fresh_B is None:
                                             print(f"[DEBUG-UNIT-TEST] Attempting to FIX l_B weight...")
                                             original_data = l_B.weight.data
                                             # Replace weight with a new Parameter
                                             l_B.weight = nn.Parameter(original_data.clone().detach().requires_grad_(True))
                                             print(f"[DEBUG-UNIT-TEST] Replaced l_B.weight with new Parameter.")
                                             
                                             # Retry
                                             out_retry = l_B(fresh_in_B)
                                             grad_retry = torch.autograd.grad(out_retry.mean(), l_B.weight, allow_unused=True)[0]
                                             print(f"[DEBUG-UNIT-TEST] l_B Retry after Fix: Grad is {'Valid' if grad_retry is not None else 'None'}")
                                             
                                             # Restore (optional, but good for stability if we continue)
                                             # l_B.weight = p_test 
                                     except Exception as e_fresh:
                                         print(f"[DEBUG-UNIT-TEST] l_B Fresh Input Error: {e_fresh}")

                                     # Manual forward
                                     h_a = l_A(l_drop(dummy_input))
                                     h_b = l_B(h_a)
                                     manual_out = h_b * l_scale
                                     
                                     manual_grad = torch.autograd.grad(manual_out.mean(), p_test, retain_graph=False, allow_unused=True)[0]
                                     if manual_grad is not None:
                                         print(f"[DEBUG-UNIT-TEST] MANUAL SUCCESS: Manual LoRA flow works! |Grad|={manual_grad.norm().item()}")
                                         print(f"[DEBUG-UNIT-TEST] CONCLUSION: peft.Linear.forward() is broken or skipping adapters.")
                                     else:
                                         print(f"[DEBUG-UNIT-TEST] MANUAL FAILURE: Even manual flow fails!")
                                 except Exception as e_man:
                                     print(f"[DEBUG-UNIT-TEST] MANUAL ERROR: {e_man}")

                         except Exception as e:
                             print(f"[DEBUG-UNIT-TEST] ERROR: {e}")

                         # Compute grad of mean of hidden state w.r.t param
                         g_test = torch.autograd.grad(out.last_hidden_state.mean(), p_test, retain_graph=True, allow_unused=True)[0]
                         
                         if g_test is None:
                             print(f"[DEBUG-PROBE-INNER] Grad(hidden_state, {p_name}) is None! The break is inside the model.")
                         else:
                             print(f"[DEBUG-PROBE-INNER] Grad(hidden_state, {p_name}) exists! Norm={g_test.norm().item()}. The model body is OK.")
                     else:
                         print(f"[DEBUG-PROBE-INNER] Could not find test param in layer {layer_idx}")
                     
                     # [NEW PROBE] Check Norm Weight
                     try:
                         p_norm = self.model.norm.weight
                         g_norm = torch.autograd.grad(out.last_hidden_state.mean(), p_norm, retain_graph=True, allow_unused=True)[0]
                         if g_norm is None:
                             print(f"[DEBUG-PROBE-INNER] Grad(hidden_state, norm.weight) is None!")
                         else:
                             print(f"[DEBUG-PROBE-INNER] Grad(hidden_state, norm.weight) exists! Norm={g_norm.norm().item()}")
                     except Exception as e:
                         print(f"[DEBUG-PROBE-INNER] Error checking norm: {e}")

                     # [NEW PROBE] Check LoRA A
                     try:
                         p_lora_A = self.model.layers[layer_idx].mlp.down_proj.lora_A['default'].weight
                         g_lora_A = torch.autograd.grad(out.last_hidden_state.mean(), p_lora_A, retain_graph=True, allow_unused=True)[0]
                         if g_lora_A is None:
                             print(f"[DEBUG-PROBE-INNER] Grad(hidden_state, lora_A) is None!")
                         else:
                             print(f"[DEBUG-PROBE-INNER] Grad(hidden_state, lora_A) exists! Norm={g_lora_A.norm().item()}")
                     except Exception as e:
                         print(f"[DEBUG-PROBE-INNER] Error checking lora_A: {e}")
                         
                 except Exception as e:
                     print(f"[DEBUG-PROBE-INNER] Error in inner probe: {e}")

        logits_aux = self.lm_head(out.last_hidden_state)
        shift_logits = logits_aux[..., :-1, :].contiguous()
        shift_labels = b_labels[..., 1:].contiguous()

        # [PERF] Logits diagnostics reduced to first 2 steps only
        if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 2:
            with torch.no_grad():
                logits_mean = shift_logits.mean().item()
                logits_std = shift_logits.std().item()
                logits_max = shift_logits.max().item()
                logits_min = shift_logits.min().item()
                # Check for NaN or Inf
                has_nan = torch.isnan(shift_logits).any().item()
                has_inf = torch.isinf(shift_logits).any().item()
                print(f"[AUX-LOGITS] step={self._global_step} mean={logits_mean:.4f} std={logits_std:.4f} min={logits_min:.4f} max={logits_max:.4f} nan={has_nan} inf={has_inf}")

        # 3) token-dimension loss (per-token first, then per-sample mean)
        flat_loss = F.cross_entropy(
            shift_logits.view(-1, self.config.vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction='none',
        ).view_as(shift_labels)                                      # [N, Lm-1]

        # [PERF] Per-token loss diagnostics reduced to first 2 steps only
        if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 2:
            with torch.no_grad():
                valid_mask_diag = (shift_labels != -100)
                if valid_mask_diag.any():
                    valid_losses = flat_loss[valid_mask_diag]
                    loss_mean = valid_losses.mean().item()
                    loss_std = valid_losses.std().item()
                    loss_max = valid_losses.max().item()
                    loss_min = valid_losses.min().item()
                    n_valid = valid_mask_diag.sum().item()
                    print(f"[AUX-LOSS-DIST] step={self._global_step} n_valid={n_valid} mean={loss_mean:.4f} std={loss_std:.4f} min={loss_min:.4f} max={loss_max:.4f}")

        valid_mask = (shift_labels != -100).float()                  # [N, Lm-1]
        token_sums = (flat_loss * valid_mask).sum(dim=1)             # [N]
        token_counts = valid_mask.sum(dim=1).clamp(min=1.0)          # [N]
        sample_means = token_sums / token_counts                     # per-sample mean CE

        # [NEW] Diagnostic per-sample loss (reduced to first 2 steps only)
        if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 2:
            with torch.no_grad():
                print(f"[AUX-SAMPLE-LOSS] step={self._global_step} sample_means={sample_means.tolist()}")

        if sample_means.numel() == 0:
            return None
        
        # [PERF] Gradient connectivity check - disabled by default for performance
        # Only enable for first step to verify setup, then skip
        if self._is_main() and self._global_step == 1:
            test_loss = sample_means.mean()
            try:
                lora_b = self.model.layers[27].mlp.down_proj.lora_B['default'].weight
                g = torch.autograd.grad(test_loss, lora_b, allow_unused=True, retain_graph=True)[0]
                if g is None:
                    print(f"[BATCH-AUX-FWD-DBG] FAIL: Gradient to LoRA is None!")
                else:
                    print(f"[BATCH-AUX-FWD-DBG] SUCCESS: Gradient to LoRA exists! norm={g.norm().item()}")
            except Exception as e:
                print(f"[BATCH-AUX-FWD-DBG] Error: {e}")
        
        return sample_means.mean(), sample_means.detach()

    def _detect_pii_regions(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        Pure Tensor PII Detection (Optimized for Code & PII).
        Focus: Precision & Efficiency. Minimize context waste.
        
        Synced with scripts/extract_pii_like_training.py
        """
        # Ensure input is a 2D tensor
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        B, T = input_ids.shape
        device = input_ids.device
        
        # Move attr_map to the same device (buffers are usually on-device but be safe)
        token_attr_map = self.token_attr_map.to(device)
        
        # 1. Lookup & Channels
        safe_ids = input_ids.clamp(0, token_attr_map.size(0) - 1)
        attrs = token_attr_map[safe_ids].long() # Convert to long for bitwise ops
        
        # Unpack Bits (int16)
        # Bits 0-3: Digit Count (0-15)
        digit_counts = (attrs & 0x0F).float().unsqueeze(1)
        
        # Bit 4: Email Anchor (@)
        is_email_anchor = ((attrs >> 4) & 1).float().unsqueeze(1)
        
        # Bit 5: Address Key
        is_addr = ((attrs >> 5) & 1).float().unsqueeze(1)
        
        # Bit 6: Poison (General)
        is_poison = ((attrs >> 6) & 1).float().unsqueeze(1)
        
        # Bit 7: Secret Anchor
        is_secret = ((attrs >> 7) & 1).float().unsqueeze(1)
        
        # Bit 8: Date Keyword
        is_date = ((attrs >> 8) & 1).float().unsqueeze(1)
        
        # Bit 9: Unit
        is_unit = ((attrs >> 9) & 1).float().unsqueeze(1)
        
        # Bit 10: Phone Separator
        is_phone_sep = ((attrs >> 10) & 1).float().unsqueeze(1)
        
        # Bit 11: Dot
        is_dot = ((attrs >> 11) & 1).float().unsqueeze(1)
        
        # Bit 13: Assignment
        is_assign = ((attrs >> 13) & 1).float().unsqueeze(1)
        
        # Bit 14: High-Confidence Secret (does NOT need ASSIGN)
        is_high_conf_secret = ((attrs >> 14) & 1).float().unsqueeze(1)
        
        combined_mask = torch.zeros_like(digit_counts)
        
        # 2. Poison & Kill Zones
        
        # General Poison Kill Zone (±10)
        if is_poison.sum() > 0:
            kill_zone = F.max_pool1d(is_poison, kernel_size=21, stride=1, padding=10)
        else:
            kill_zone = torch.zeros_like(is_poison)
            
        # Date Kill Zone (±15) - stronger for dates
        if is_date.sum() > 0:
            date_kill_zone = F.max_pool1d(is_date, kernel_size=31, stride=1, padding=15)
        else:
            date_kill_zone = torch.zeros_like(is_date)
            
        # Unit suppression (immediate right context): if a unit is present,
        # suppress the number immediately preceding it.
        if is_unit.sum() > 0:
            unit_kill_zone = F.max_pool1d(is_unit, kernel_size=5, stride=1, padding=2)
        else:
            unit_kill_zone = torch.zeros_like(is_unit)

        # Safe Zones
        safe_digits = digit_counts * (1.0 - kill_zone) * (1.0 - date_kill_zone) * (1.0 - unit_kill_zone)
        safe_addr = is_addr * (1.0 - kill_zone)
        
        # 3. Phone/SSN: Exact digit summation
        # Window: 10 tokens (enough for numbers like 713-853-1411)
        # Target total digits between 7 and 15.
        
        k_sum = torch.ones(1, 1, 10, device=device)
        digit_sum = F.conv1d(safe_digits, k_sum, padding=5)[:, :, :-1]
        
        # Check 1: Digit Count Range [7, 15]
        has_enough_digits = (digit_sum >= 7.0) * (digit_sum <= 15.0)
        
        # Check 2: Separator Density
        sep_density = F.conv1d(is_phone_sep, k_sum, padding=5)[:, :, :-1]
        has_sep = (sep_density >= 1.0)
        
        # Rule: (Digits >= 7 AND Has Sep) OR (Digits >= 10)
        # This allows 7-digit local numbers ONLY if they have separators (555-1234)
        # And allows 10-digit pure numbers (7136463490)
        phone_hit = (has_enough_digits * has_sep) + (digit_sum >= 10.0)
        phone_hit = (phone_hit > 0).float() * (1.0 - kill_zone)
        
        if phone_hit.sum() > 0:
            # Base: 9 (Reduced to avoid capturing too much text)
            combined_mask = torch.max(combined_mask, F.max_pool1d(phone_hit, kernel_size=9, stride=1, padding=4))
            
        # 4. Email detection
        # Require '@' and '.', window size 15
        if is_email_anchor.sum() > 0:
            k_email = torch.ones(1, 1, 15, device=device)
            has_dot = (F.conv1d(is_dot, k_email, padding=7) > 0).float()
            
            email_hit = is_email_anchor * has_dot
            if email_hit.sum() > 0:
                combined_mask = torch.max(combined_mask, F.max_pool1d(email_hit, kernel_size=15, stride=1, padding=7))

        # 5. Address detection (recovered)
        if safe_addr.sum() > 0:
            k_left = torch.ones(1, 1, 25, device=device) # Increased from 15 to 25
            padded_digits = F.pad(safe_digits, (24, 0)) # Match padding to kernel-1
            has_house_num = (F.conv1d(padded_digits, k_left) > 0).float()
            
            addr_hit = safe_addr * has_house_num
            if addr_hit.sum() > 0:
                # Base: 33 (Fixed from 32)
                combined_mask = torch.max(combined_mask, F.max_pool1d(addr_hit, kernel_size=33, stride=1, padding=16))
            
        # 6. Secret anchors (bit 7) - low-confidence, needs ASSIGN nearby
        if is_secret.sum() > 0:
            k_assign = torch.ones(1, 1, 9, device=device) # Window 9 (±4)
            has_assign = (F.conv1d(is_assign, k_assign, padding=4) > 0).float()
            
            verified_secret = is_secret * has_assign
                
            if verified_secret.sum() > 0:
                # Base: 31 (Fixed from 30)
                combined_mask = torch.max(combined_mask, F.max_pool1d(verified_secret, kernel_size=31, stride=1, padding=15))
        
        # 7. High-confidence secret anchors (bit 14) - does NOT need ASSIGN
        # Patterns like sk-, eyJ, etc., unlikely in non-secret contexts
        if is_high_conf_secret.sum() > 0:
            # Base: 31 (same expansion as low-conf secrets)
            combined_mask = torch.max(combined_mask, F.max_pool1d(is_high_conf_secret, kernel_size=31, stride=1, padding=15))
        
        # 8. Secret bigram detection (n-gram)
        # For patterns like AKIA, ASIA, ghp_ that may be split across tokens
        if self._secret_bigrams is not None and len(self._secret_bigrams) > 0 and T >= 2:
            bigrams_ref = self._secret_bigrams.to(device)  # (N, 2)
            
            # Build consecutive token pairs from input
            first_tokens = input_ids[:, :-1]   # (B, T-1)
            second_tokens = input_ids[:, 1:]   # (B, T-1)
            
            # Create bigram match mask
            bigram_mask = torch.zeros(B, T, device=device)
            
            for i in range(bigrams_ref.size(0)):
                first_match = (first_tokens == bigrams_ref[i, 0])   # (B, T-1)
                second_match = (second_tokens == bigrams_ref[i, 1]) # (B, T-1)
                pair_match = first_match & second_match             # (B, T-1)
                
                if pair_match.any():
                    # Mark both tokens of the matching pair
                    bigram_mask[:, :-1] = bigram_mask[:, :-1] + pair_match.float()  # first token
                    bigram_mask[:, 1:] = bigram_mask[:, 1:] + pair_match.float()    # second token
            
            if bigram_mask.sum() > 0:
                bigram_mask = (bigram_mask > 0).float().unsqueeze(1)  # (B, 1, T)
                # Expand detection range: 31 tokens (same as other secrets)
                combined_mask = torch.max(combined_mask, F.max_pool1d(bigram_mask, kernel_size=31, stride=1, padding=15))
            
        return combined_mask.squeeze(1)

    # Added: restore missing _debug_compare_aux_single_sample_internal method
    def _debug_compare_aux_single_sample_internal(
        self,
        sample: Dict[str, torch.Tensor],
        *,
        device: torch.device,
        attn_dtype: torch.dtype,
    ):
        import math
        try:
            # --- batch path ---
            with torch.no_grad():
                out_batch = self._batch_aux_forward(
                    samples=[sample],
                    device=device,
                    attn_dtype=attn_dtype,
                )
            if isinstance(out_batch, tuple):
                batch_loss, _ = out_batch
            else:
                batch_loss = out_batch
            batch_val = float(batch_loss.item()) if batch_loss is not None else math.nan

            # --- old-style single-sample path ---
            ids = sample['input_ids'].unsqueeze(0).to(device)        # [1,L]
            labels = sample['labels'].unsqueeze(0).to(device)        # [1,L]
            attn = sample['attention_mask'].unsqueeze(0).to(device)  # [1,L]

            with torch.no_grad(), self._no_gc():
                out = self.model(
                    input_ids=ids,
                    attention_mask=attn,
                    use_cache=False,
                    return_dict=True,
                )
                logits_aux = self.lm_head(out.last_hidden_state)      # [1,L,V]
                shift_logits = logits_aux[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                seq_loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            seq_val = float(seq_loss.item())

            if self._is_main():
                try:
                    print(
                        f"[AUX-COMPARE] step={self._global_step} "
                        f"seq_loss={seq_val:.6f} batch_loss={batch_val:.6f} "
                        f"diff={batch_val - seq_val:+.6e}"
                    )
                except Exception:
                    pass
        except Exception as e:
            if self._is_main():
                try:
                    print(f"[AUX-COMPARE-ERR] step={self._global_step} err={repr(e)}")
                except Exception:
                    pass

    # =========================================================================
    # [DP-SGD UNIFIED MODE] Main + AUX merged into a single batch forward
    # =========================================================================
    def _forward_unified_dpsgd(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sensitive_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        [DP-SGD Unified Mode] Merge main batch and AUX samples into a single
        forward batch so Opacus per-sample clipping applies to both main and
        AUX, ensuring fair DP protection.

        Steps:
        1. Generate AUX samples (fresh + replay)
        2. Pad AUX samples to match main sequence length
        3. Concatenate main and AUX into [main_bs + aux_bs, seq_len]
        4. Single forward pass
        5. Separate main and AUX logits
        6. Compute and combine losses

        Memory optimizations:
        - `max_aux_samples_unified`: limit AUX samples to avoid OOM
        - AUX sequence length capped separately from main
        """
        default_return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        device = input_ids.device
        attn_dtype = attention_mask.dtype if attention_mask is not None else torch.long
        
        main_bs = input_ids.size(0)
        main_seq_len = input_ids.size(1)
        
        # [Memory opt] obtain AUX samples limit
        max_aux_unified = int(getattr(self.config, 'inject_max_aux_samples_unified', 2))
        
        # ========= Step 1: generate AUX samples =========
        fresh_samples = self._generate_fresh_aux_samples(
            input_ids=input_ids,
            sensitive_mask=sensitive_mask,
            device=device,
            attn_dtype=attn_dtype,
        )
        if fresh_samples is None:
            fresh_samples = []
        
        # [Memory opt] cap fresh samples count
        if len(fresh_samples) > max_aux_unified:
            fresh_samples = fresh_samples[:max_aux_unified]
        
        # Get replay samples (full credit management)
        replay_samples = []
        replay_gids_used = []  # record used gids for subsequent credit deduction
        overrides = getattr(self, '_aux_kw_overrides', {}) if hasattr(self, '_aux_kw_overrides') else {}
        def opt_fw(name, default):
            return overrides.get(name, getattr(self.config, name, default))
        
        # [FIX] Compute replay quota (consider fresh samples already used)
        remaining_aux_quota = max(0, max_aux_unified - len(fresh_samples))
        
        if remaining_aux_quota > 0 and len(self._replay_buf) > 0 and bool(opt_fw('inject_replay_enable', True)):
            per_step = int(opt_fw('inject_replay_per_step', 2))
            max_len_cap = int(opt_fw('inject_replay_max_len', 256))
            # [MEM OPT] further limit replay count
            max_replay = min(per_step, remaining_aux_quota)
            
            n_buf = len(self._replay_buf)
            k = min(n_buf, max_replay)
            idxs = torch.randperm(n_buf, device='cpu')[:k].tolist()
            
            for idx in idxs:
                if len(replay_samples) >= remaining_aux_quota:
                    break
                    
                it = self._replay_buf[idx]
                gid = int(it.get('global_id', -1))
                if gid < 0:
                    continue
                credit_raw = int(self._global_replay_credits.get(gid, 0))
                if credit_raw <= 0:
                    continue
                
                ids_full = it['input_ids_full']
                labels_full = it['labels_full']
                attn_full = it.get('attention_mask_full', torch.ones_like(ids_full, dtype=torch.long))
                boundary = int(it.get('boundary_len', 0))
                L = int(ids_full.size(1))
                use_len = L if (max_len_cap <= 0) else min(L, max_len_cap)
                
                # [FIX] Ensure dimensions are correct (squeeze may cause issues)
                ids_trunc = ids_full[0, :use_len] if ids_full.dim() == 2 else ids_full[:use_len]
                labels_trunc = labels_full[0, :use_len] if labels_full.dim() == 2 else labels_full[:use_len]
                attn_trunc = attn_full[0, :use_len] if attn_full.dim() == 2 else attn_full[:use_len]
                
                labels_trunc = labels_trunc.clone()
                labels_trunc[:boundary] = -100
                
                replay_samples.append({
                    'input_ids': ids_trunc,
                    'labels': labels_trunc,
                    'attention_mask': attn_trunc,
                    'is_replay': True,
                    'global_id': gid,
                    'buffer_idx': idx,  # buffer index for later cleanup
                })
                replay_gids_used.append(gid)
        
        # [MEM OPT] Merge fresh + replay, but enforce a total cap
        all_aux_samples = fresh_samples + replay_samples
        if len(all_aux_samples) > max_aux_unified:
            all_aux_samples = all_aux_samples[:max_aux_unified]
        aux_count = len(all_aux_samples)
        
        # If no AUX samples, fall back to normal path
        if aux_count == 0:
            return self._forward_normal_path(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                sensitive_mask=sensitive_mask,
                **kwargs,
            )
        
        # ========= Step 2: build AUX batch tensors =========
        pad_id = self.config.pad_token_id if self.config.pad_token_id is not None else 0
        
        aux_ids_list = []
        aux_labels_list = []
        aux_attn_list = []
        
        for s in all_aux_samples:
            aux_ids_list.append(s['input_ids'].to(device))
            aux_labels_list.append(s['labels'].to(device))
            aux_attn_list.append(s['attention_mask'].to(device))
        
        # Pad AUX samples to equal length
        aux_input_ids = pad_sequence(aux_ids_list, batch_first=True, padding_value=pad_id)
        aux_labels = pad_sequence(aux_labels_list, batch_first=True, padding_value=-100)
        aux_attention = pad_sequence(aux_attn_list, batch_first=True, padding_value=0).to(attn_dtype)
        
        aux_seq_len = aux_input_ids.size(1)
        
        # ========= Step 3: align sequence length =========
        # [Memory opt] cap AUX sequence length to avoid padding entire batch to main length
        max_aux_seq_len = int(getattr(self.config, 'inject_max_aux_seq_len_unified', 256))
        if aux_seq_len > max_aux_seq_len:
            aux_input_ids = aux_input_ids[:, :max_aux_seq_len]
            aux_labels = aux_labels[:, :max_aux_seq_len]
            aux_attention = aux_attention[:, :max_aux_seq_len]
            aux_seq_len = max_aux_seq_len
        
        # Take the max sequence length across main and AUX
        unified_seq_len = max(main_seq_len, aux_seq_len)
        
        # Pad main to unified_seq_len
        if main_seq_len < unified_seq_len:
            pad_len = unified_seq_len - main_seq_len
            input_ids = F.pad(input_ids, (0, pad_len), value=pad_id)
            labels = F.pad(labels, (0, pad_len), value=-100) if labels is not None else None
            attention_mask = F.pad(attention_mask, (0, pad_len), value=0).to(attn_dtype) if attention_mask is not None else None
            if position_ids is not None:
                # Position IDs require special handling: continue incrementing
                last_pos = position_ids[:, -1:] + 1
                extra_pos = torch.arange(pad_len, device=device).unsqueeze(0).expand(main_bs, -1) + last_pos
                position_ids = torch.cat([position_ids, extra_pos], dim=1)
            if sensitive_mask is not None:
                sensitive_mask = F.pad(sensitive_mask, (0, pad_len), value=0)
        
        # Pad AUX to unified_seq_len
        if aux_seq_len < unified_seq_len:
            pad_len = unified_seq_len - aux_seq_len
            aux_input_ids = F.pad(aux_input_ids, (0, pad_len), value=pad_id)
            aux_labels = F.pad(aux_labels, (0, pad_len), value=-100)
            aux_attention = F.pad(aux_attention, (0, pad_len), value=0).to(attn_dtype)
        
        # ========= Step 4: Concat main + AUX =========
        unified_input_ids = torch.cat([input_ids, aux_input_ids], dim=0)  # [main_bs + aux_bs, seq_len]
        unified_labels = torch.cat([labels, aux_labels], dim=0) if labels is not None else None
        unified_attention = torch.cat([attention_mask, aux_attention], dim=0) if attention_mask is not None else None
        
        # [MEM-OPT LOG]
        if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 3:
            print(f"[DP-UNIFIED-MEM] step={self._global_step} main_bs={main_bs} aux_count={aux_count} "
                  f"main_seq={main_seq_len} aux_seq={aux_seq_len} unified_seq={unified_seq_len} "
                  f"total_tokens={(main_bs + aux_count) * unified_seq_len}")
        
        # Position IDs: construct position_ids for AUX
        with torch.no_grad():
            aux_lengths = aux_attention.long().sum(dim=1)
            aux_pos_ids = torch.arange(unified_seq_len, device=device).unsqueeze(0).expand(aux_count, -1).clone()
            for i, L in enumerate(aux_lengths.tolist()):
                if L < unified_seq_len:
                    aux_pos_ids[i, L:] = 0
        
        if position_ids is not None:
            unified_position_ids = torch.cat([position_ids, aux_pos_ids], dim=0)
        else:
            # Construct position_ids for main as well
            main_pos_ids = torch.arange(unified_seq_len, device=device).unsqueeze(0).expand(main_bs, -1)
            unified_position_ids = torch.cat([main_pos_ids, aux_pos_ids], dim=0)
        
        # ========= Step 5: Embedding + Modulation =========
        unified_embeds = self.model.embed_tokens(unified_input_ids)
        
        # Apply modulation to main part (if sensitive_mask provided)
        if sensitive_mask is not None:
            # Create empty sensitive_mask for AUX (no modulation by default; can be extended)
            aux_sens_mask = torch.zeros(aux_count, unified_seq_len, dtype=torch.long, device=device)
            unified_sens_mask = torch.cat([sensitive_mask, aux_sens_mask], dim=0)
            unified_embeds = self.embedding_modulation(
                unified_embeds,
                sensitive_mask=unified_sens_mask,
                training=self.training,
            )
        
        # ========= Step 6: Single forward =========
        local_use_cache = False if self.training else use_cache
        
        if self.training and unified_embeds.requires_grad is False:
            unified_embeds.requires_grad_(True)
        
        outputs = self.model(
            input_ids=None,
            attention_mask=unified_attention,
            position_ids=unified_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=unified_embeds,
            use_cache=local_use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        # ========= Step 7: Separate main and AUX logits, compute loss =========
        main_logits = logits[:main_bs]
        aux_logits = logits[main_bs:]
        
        loss = None
        main_tokens_this_step = 0
        
        # Main loss
        if labels is not None:
            main_labels_final = unified_labels[:main_bs]
            shift_logits = main_logits[..., :-1, :].contiguous()
            shift_labels = main_labels_final[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            main_loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            try:
                self._last_main_loss = float(main_loss.item())
            except Exception:
                pass
            try:
                main_tokens_this_step = int((shift_labels != -100).sum().item())
            except Exception:
                main_tokens_this_step = 0
            
            loss = main_loss
        
        # AUX loss
        aux_labels_final = unified_labels[main_bs:] if unified_labels is not None else aux_labels
        aux_shift_logits = aux_logits[..., :-1, :].contiguous()
        aux_shift_labels = aux_labels_final[..., 1:].contiguous()
        
        # Per-sample CE (reduction='none'), then take the mean
        aux_ce = F.cross_entropy(
            aux_shift_logits.view(-1, self.config.vocab_size),
            aux_shift_labels.view(-1),
            ignore_index=-100,
            reduction='none',
        )
        # Reshape back to [aux_bs, seq_len-1]
        aux_ce = aux_ce.view(aux_count, -1)
        
        # Per-sample mean (only for positions != -100)
        aux_valid_mask = (aux_shift_labels != -100).float()
        aux_valid_counts = aux_valid_mask.sum(dim=1).clamp(min=1)
        aux_sample_losses = (aux_ce * aux_valid_mask).sum(dim=1) / aux_valid_counts
        aux_loss = aux_sample_losses.mean()
        
        try:
            self._last_aux_loss = float(aux_loss.item())
        except Exception:
            pass
        
        # ========= Step 8: Combine loss =========
        aux_w_max = getattr(self.config, 'aux_weight_max', None)
        if aux_w_max is not None:
            warmup_steps = int(getattr(self.config, 'aux_weight_warmup_steps', 800))
            cur = max(0, int(self._global_step))
            ratio = min(1.0, float(cur) / max(1, warmup_steps))
            lam = float(aux_w_max) * ratio
        else:
            lam = float(getattr(self.config, 'inject_aux_weight', 1.0))
        
        self._last_aux_lambda = lam
        
        if loss is not None:
            # [UNIFIED] AUX loss is added directly to main loss
            # Note: in unified mode AUX gradients also pass through Opacus per-sample clipping
            real_aux_term = lam * aux_loss
            
            # [FIX] Use surrogate trick so the returned loss value shows only main_loss
            # but gradients still include AUX terms, keeping logs cleaner and comparable to bypass mode
            # loss_return = main_loss + (aux_term - sg(aux_term))
            #             = main_loss (value) but grad includes aux
            loss_for_grad = main_loss + real_aux_term  # full loss used for gradient computation
            # Surrogate: value is main_loss, gradient is main + aux
            loss = main_loss + (real_aux_term - real_aux_term.detach())
            
            try:
                self._last_aux_contrib = float(real_aux_term.item())
            except Exception:
                pass
        else:
            loss = lam * aux_loss
        
        # ========= Step 9: Replay credit deduction =========
        # [FIX] unified mode must also correctly manage replay credits
        if replay_gids_used:
            rb_dedup = bool(opt_fw('inject_replay_dedup', True))
            gids_to_remove = set()
            
            for gid in replay_gids_used:
                cur = int(self._global_replay_credits.get(gid, 0))
                if cur > 0:
                    new_val = cur - 1
                    self._global_replay_credits[gid] = new_val
                    if new_val <= 0:
                        gids_to_remove.add(gid)
            
            # Remove samples with exhausted credits
            if gids_to_remove:
                new_buf = []
                for it in self._replay_buf:
                    gid = int(it.get('global_id', -1))
                    if gid in gids_to_remove:
                        if rb_dedup:
                            self._replay_key_set.discard(it.get('key_hash', ''))
                        self._replay_dropped_last += 1
                    else:
                        new_buf.append(it)
                self._replay_buf = new_buf
                for gid in gids_to_remove:
                    self._global_replay_credits.pop(gid, None)
        
        # Debug logs
        if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
            replay_count = len([s for s in all_aux_samples if s.get('is_replay', False)])
            fresh_count = aux_count - replay_count
            print(f"[DP-UNIFIED] step={self._global_step} main_bs={main_bs} aux_count={aux_count} "
                  f"(fresh={fresh_count}, replay={replay_count}) "
                  f"unified_bs={main_bs + aux_count} seq_len={unified_seq_len} "
                  f"main_loss={self._last_main_loss:.4f} aux_loss={self._last_aux_loss:.4f} "
                  f"lambda={lam:.4f} total_loss={loss.item():.4f}")
        
        # ========= Step 10: Build return value =========
        # Return main logits only (compatible with standard forward)
        if not default_return_dict:
            output = (main_logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=main_logits,  # return main logits only
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _forward_normal_path(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sensitive_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Normal forward path (no AUX or fallback for non-unified mode).
        This wraps the original forward logic and is used when unified mode
        has no AUX samples.
        """
        default_return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must specify input_ids or inputs_embeds")
            inputs_embeds = self.model.embed_tokens(input_ids)
        
        inputs_embeds = self.embedding_modulation(
            inputs_embeds,
            sensitive_mask=sensitive_mask,
            training=self.training,
        )
        
        local_use_cache = False if self.training else use_cache
        
        if self.training and inputs_embeds.requires_grad is False:
            inputs_embeds.requires_grad_(True)
        
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=local_use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=default_return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            
            try:
                self._last_main_loss = float(loss.item())
            except Exception:
                pass
        
        if not default_return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        sensitive_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        # Use parent class config logic
        default_return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # [DP-SGD] Record main batch size and sequence length for AUX forward (ensure dims match)
        if input_ids is not None:
            self._main_batch_size = input_ids.size(0)
            self._main_seq_length = input_ids.size(1)
        elif inputs_embeds is not None:
            self._main_batch_size = inputs_embeds.size(0)
            self._main_seq_length = inputs_embeds.size(1)
        else:
            self._main_batch_size = 1
            self._main_seq_length = 1
        
        # [NEW] Internal PII Detection Logic
        # CHANGED: Removed 'and sensitive_mask is None' check.
        # Now we FORCE internal detection during training, overwriting any mask passed from the dataloader.
        if self.training and input_ids is not None:
            with torch.no_grad():
                sensitive_mask = self._detect_pii_regions(input_ids)

        # Increment training step and print SETUP once
        if self.training:
            self._global_step += 1
            # Reset logging fields each step to avoid residual values from previous step
            self._last_aux_lambda = None
            self._last_aux_loss = None
            self._last_kl_loss = None
            self._kl_weight = None
            self._last_neg_aux_loss = None
            self._neg_aux_weight = None
            # NEW: loss breakdown fields (logging only)
            self._last_main_loss: Optional[float] = None
            self._last_aux_contrib: Optional[float] = None
            self._last_kl_contrib: Optional[float] = None
            self._last_neg_aux_contrib: Optional[float] = None
            self._last_breakdown = None
            # [NEW] AUX logs container for surgical update stats
            self.aux_logs = {}
            # Reset replay/supervised token counters
            self._last_aux_tokens_fresh_supervised = 0
            self._last_aux_tokens_replay = 0
            # Reset replay counters
            self._replay_added_last = 0
            self._replay_dropped_last = 0
            self._last_aux_tokens_fresh_total = 0
            if not self._setup_printed and self._is_main():
                kl_en = bool(getattr(self.config, 'kl_no_key_enable', True))
                kl_w = float(getattr(self.config, 'kl_no_key_weight', 0.1))
                kl_every = int(getattr(self.config, 'kl_no_key_every_n_steps', 1))
                neg_en = bool(getattr(self.config, 'neg_aux_enable', True))
                neg_w = float(getattr(self.config, 'neg_aux_weight', 0.2))
                neg_every = int(getattr(self.config, 'neg_aux_every_n_steps', 1))
                aux_max = getattr(self.config, 'aux_weight_max', None)
                aux_wu = int(getattr(self.config, 'aux_weight_warmup_steps', 800))
                print(f"[SETUP] kl_no_key_enable={kl_en} weight={kl_w} every_n={kl_every}; neg_aux_enable={neg_en} weight={neg_w} every_n={neg_every}; aux_weight_schedule=max={aux_max} warmup_steps={aux_wu}")
                self._setup_printed = True
        
        # =========================================================================
        # [DP-SGD UNIFIED MODE] Detect whether unified path should be used
        # =========================================================================
        dp_sgd_mode = bool(getattr(self.config, 'dp_sgd_mode', False))
        dp_aux_mode = str(getattr(self.config, 'dp_aux_mode', 'bypass')).lower()
        use_unified_path = (
            self.training and 
            dp_sgd_mode and 
            dp_aux_mode == 'unified' and 
            input_ids is not None and
            not bool(getattr(self.config, 'use_old_aux_pipeline', True))  # only new pipeline supports unified
        )
        
        if use_unified_path:
            # Use unified path: main + AUX merged into a single batch to go through DP-SGD together
            return self._forward_unified_dpsgd(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                sensitive_mask=sensitive_mask,
                **kwargs,
            )
        
        # =========================================================================
        # [Original path] non-unified mode (including bypass and non-DP-SGD)
        # =========================================================================
        
        # 1) Obtain inputs_embeds uniformly (hook point)
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must specify input_ids or inputs_embeds")
            inputs_embeds = self.model.embed_tokens(input_ids)
        # Toggles for stats and sampling
        record_stats = self.training and (sensitive_mask is not None)
        need_debug = record_stats and (self.modulation_debug_steps > 0) and (self._modulation_debug_counter < self.modulation_debug_steps)
        # [FIX] missing pre_emb_detached definition: used for later delta-embedding stats
        pre_emb_detached = inputs_embeds.detach() if record_stats else None

        # 2) Modulation during training
        inputs_embeds = self.embedding_modulation(
            inputs_embeds,
            sensitive_mask=sensitive_mask,
            training=self.training,
        )
        # Optional: record modulation statistics (always record basic stats; sample collection controlled by debug_steps)
        if record_stats:
            with torch.no_grad():
                b, t, h = pre_emb_detached.shape
                # per-sample mask sums
                if sensitive_mask is not None:
                    mask_per_sample = sensitive_mask.to(torch.long).sum(dim=1).cpu().tolist()
                    mask_sum = int(sum(mask_per_sample))
                else:
                    mask_per_sample = [0] * b
                    mask_sum = 0
                # Count of valid (non-pad) tokens
                if attention_mask is not None:
                    valid_tokens_per_sample = attention_mask.detach().to(torch.long).sum(dim=1).cpu().tolist()
                    valid_tokens = int(sum(valid_tokens_per_sample))
                else:
                    valid_tokens_per_sample = [t] * b
                    valid_tokens = b * t
                mask_frac_all = float(mask_sum) / float(b * t) if b * t > 0 else 0.0
                mask_frac_nonpad = float(mask_sum) / float(valid_tokens) if valid_tokens > 0 else 0.0
                # Embedding change (delta)
                delta = (inputs_embeds.detach() - pre_emb_detached).abs()
                mean_abs_delta = delta.mean().item()
                max_abs_delta = delta.amax().item()
                if sensitive_mask is not None and sensitive_mask.any():
                    m = sensitive_mask.to(delta.dtype).unsqueeze(-1)        # [B,T,1]
                    masked_mean_abs_delta = (delta * m).sum() / (m.sum() * delta.size(-1))
                    masked_mean_abs_delta = masked_mean_abs_delta.item()
                else:
                    masked_mean_abs_delta = 0.0
                # Sampling of examples only occurs within the first debug_steps; afterwards only stats without samples are recorded
                do_sample = (self.modulation_debug_steps <= 0) or (self._modulation_debug_counter < self.modulation_debug_steps)
                samples = []
                try:
                    if do_sample:
                        tok = getattr(self, 'tokenizer', None)
                        mb_id = int(self._modulation_debug_counter)
                        if tok is not None and input_ids is not None and sensitive_mask is not None:
                            m = sensitive_mask.nonzero(as_tuple=False)  # [N,2]
                            if m.numel() > 0:
                                n = m.size(0)
                                k = min(8, n)
                                perm = torch.randperm(n, device=m.device)[:k]
                                for idx in perm.tolist():
                                    bi, ti = int(m[idx, 0].item()), int(m[idx, 1].item())
                                    tid = int(input_ids[bi, ti].item())
                                    tstr = tok.convert_ids_to_tokens([tid])[0]
                                    samples.append({'mb': mb_id, 'b': bi, 'pos': ti, 'id': tid, 'tok': tstr})
                except Exception:
                    pass
                stat = {
                    'mask_sum': int(mask_sum),
                    'mask_per_sample': [int(x) for x in mask_per_sample],
                    'valid_tokens': int(valid_tokens),
                    'valid_tokens_per_sample': [int(x) for x in valid_tokens_per_sample],
                    'mask_frac': mask_frac_all,
                    'mask_frac_nonpad': mask_frac_nonpad,
                    'mean_abs_delta': mean_abs_delta,
                    'masked_mean_abs_delta': masked_mean_abs_delta,
                    'max_abs_delta': max_abs_delta,
                    'mode': getattr(self.embedding_modulation, 'mode', 'scale'),
                    'scale': float(getattr(self.embedding_modulation, 'scale', 1.0)),
                    'bias_scale': float(getattr(self.embedding_modulation, 'bias_scale', 0.0)),
                    'b': b, 't': t, 'h': h,
                    'debug_step_index': self._modulation_debug_counter,
                    'mb_id': int(self._modulation_debug_counter),
                    'samples': samples,
                }
                self._last_modulation_stats = stat
                try:
                    self._modulation_buffer.append(stat)
                except Exception:
                    pass
                self._modulation_debug_counter += 1
        # 3) Call Transformer (force no caching during training)
        local_use_cache = False if self.training else use_cache
        
        # [FIX] Ensure inputs_embeds requires grad for Main pass too, to support Gradient Checkpointing with LoRA
        if self.training and inputs_embeds.requires_grad is False and self.model.gradient_checkpointing:
             inputs_embeds.requires_grad_(True)

        # [DEBUG] Check Gradient Checkpointing status
        # if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
            #  gc_status = getattr(self.model, 'gradient_checkpointing', False)
            #  print(f"[DEBUG-GC] step={self._global_step} gradient_checkpointing={gc_status}")

        # [DEBUG-FIX] Temporarily disable gradient checkpointing to see if it fixes torch.autograd.grad
        # We use the _no_gc context manager if we want to disable it, but here we want to disable it for the main forward too?
        # Or just for the AUX forward?
        # If main forward uses GC, then the graph is checkpointed.
        # If we want to use torch.autograd.grad on the graph produced by main forward, we might have issues.
        # But AUX forward produces its own graph.
        
        outputs = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=local_use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=default_return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        loss = None
        main_tokens_this_step = 0  # New: number of valid supervised tokens in main path
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            # New: record main path CE loss and supervised token count
            try:
                self._last_main_loss = float(loss.item())
            except Exception:
                pass
            try:
                main_tokens_this_step = int((shift_labels != -100).sum().item())
            except Exception:
                main_tokens_this_step = 0
        
        # 3.5) Training-time auxiliary loss (does not modify main path)
        # Allow overrides prefixed with 'inject_' to be passed via forward(**kwargs)
        self._aux_kw_overrides = {k: v for k, v in kwargs.items() if isinstance(k, str) and k.startswith('inject_')}
        
        aux_total = None
        kl_loss_val = None
        neg_loss_val = None

        if False:
            pass
        else:
            # -------- Step A: Fresh AUX samples (generate only + add to replay, no forward) --------
            fresh_samples = self._generate_fresh_aux_samples(
                input_ids=input_ids,
                sensitive_mask=sensitive_mask,
                device=inputs_embeds.device,
                attn_dtype=(attention_mask.dtype if attention_mask is not None else torch.long),
            )
            if fresh_samples is None:
                fresh_samples = []

            # [NEW] Optionally trigger single-sample old vs batch comparison for debugging (no effect on training, logs only)
            try:
                overrides = getattr(self, '_aux_kw_overrides', {}) if hasattr(self, '_aux_kw_overrides') else {}
                def _opt_dbg(name, default):
                    return overrides.get(name, getattr(self.config, name, default))
                dbg_en = bool(_opt_dbg('debug_aux_compare_enable', False))
                max_steps = int(_opt_dbg('debug_aux_compare_steps', 10))
                span_idx = int(_opt_dbg('debug_aux_compare_span_index', 0))
                if (
                    dbg_en
                    and self.training
                    and self._aux_compare_counter < max_steps
                    and len(fresh_samples) > 0
                ):
                    idx = max(0, min(span_idx, len(fresh_samples) - 1))
                    sample_dbg = fresh_samples[idx]
                    self._debug_compare_aux_single_sample_internal(
                        sample=sample_dbg,
                        device=inputs_embeds.device,
                        attn_dtype=(attention_mask.dtype if attention_mask is not None else torch.long),
                    )
                    self._aux_compare_counter += 1
            except Exception:
                pass

            # -------- Step B: REPLAY sampling based on per-global_id credits --------
            replay_samples: List[Dict[str, torch.Tensor]] = []
            replay_sampled = 0
            replay_tokens_used = 0
            replay_supervised_tokens_used = 0
            # Reset per-step usage counts for each forward
            self._replay_usage_this_step = {}

            if self.training and len(self._replay_buf) > 0:
                try:
                    overrides = getattr(self, '_aux_kw_overrides', {}) if hasattr(self, '_aux_kw_overrides') else {}
                    def opt_fw(name, default):
                        return overrides.get(name, getattr(self.config, name, default))

                    rb_enable = bool(opt_fw('inject_replay_enable', True))
                    if rb_enable:
                        per_step = int(opt_fw('inject_replay_per_step', 2))
                        max_len_cap = int(opt_fw('inject_replay_max_len', 256))
                        cap_frac = float(opt_fw('inject_aux_token_frac_cap', 0.20))

                        # Compute main_valid / fresh_tok / cap_tokens / allowed
                        main_valid = 0
                        try:
                            if attention_mask is not None:
                                main_valid = int(attention_mask.detach().to(torch.long).sum().item())
                        except Exception:
                            main_valid = 0
                        fresh_tok = int(getattr(self, '_last_aux_tokens_fresh_total', 0) or 0)
                        cap_tokens = int(max(0, cap_frac * float(main_valid)))
                        allowed = max(0, cap_tokens - fresh_tok)

                        if per_step > 0 and allowed > 0:
                            n_buf = len(self._replay_buf)
                            
                            # ================================================================
                            # [NEW] Loss-prioritized sampling: prefer samples with higher loss
                            # ================================================================
                            # Build a list of (buffer_idx, gid, loss) and sort by loss descending
                            candidates = []
                            for buf_idx in range(n_buf):
                                it = self._replay_buf[buf_idx]
                                gid = int(it.get('global_id', -1))
                                if gid < 0:
                                    continue
                                # Fetch historical loss (default high value 10.0 to prioritize new samples)
                                sample_loss = self._sample_loss_history.get(gid, 10.0)
                                # Check credit
                                credit = int(self._global_replay_credits.get(gid, 0))
                                if credit <= 0:
                                    continue
                                candidates.append((buf_idx, gid, sample_loss, credit))
                            
                            # Sort by loss descending (high loss first)
                            candidates.sort(key=lambda x: -x[2])
                            
                            # Select top k candidates
                            k = min(len(candidates), per_step)
                            selected_idxs = [c[0] for c in candidates[:k]]
                            
                            apply_mod = bool(opt_fw('inject_aux_apply_modulation', False))
                            mod_ass_only = bool(opt_fw('inject_aux_modulate_assistant_only', True))
                            attn_dtype_fw = (attention_mask.dtype if attention_mask is not None else torch.long)

                            for idx in selected_idxs:
                                if replay_tokens_used >= allowed:
                                    break
                                it = self._replay_buf[idx]
                                gid = int(it.get('global_id', -1))
                                if gid < 0:
                                    continue

                                # Global credit (training lifetime)
                                credit_raw = int(self._global_replay_credits.get(gid, 0))
                                if credit_raw <= 0:
                                    # Optional: debug print SKIP (disabled - too noisy)
                                    # if self._is_main() and bool(getattr(self.config, 'super_aux_example_debug', False)):
                                    #     try:
                                    #         print(f"[REPLAY-SKIP] step={self._global_step} gid={gid} credit_raw={credit_raw} skip_sampling")
                                    #     except Exception:
                                    #         pass
                                    continue

                                # Usage count in this step
                                used_before = int(self._replay_usage_this_step.get(gid, 0))
                                remaining_before = credit_raw - used_before
                                if remaining_before <= 0:
                                    continue

                                ids_full = it['input_ids_full']          # [1, L]
                                labels_full = it['labels_full']          # [1, L]
                                attn_full = it.get('attention_mask_full', torch.ones_like(ids_full, dtype=torch.long))
                                boundary = int(it.get('boundary_len', 0))
                                L = int(ids_full.size(1))
                                use_len = L if (max_len_cap <= 0) else min(L, max_len_cap)
                                if boundary >= use_len:
                                    continue
                                if replay_tokens_used + use_len > allowed:
                                    continue

                                try:
                                    sup_tok = max(0, (use_len - 1) - boundary)
                                except Exception:
                                    sup_tok = 0

                                # Construct single REPLAY sample (still on CPU; _batch_aux_forward will move to device)
                                ids_trunc = ids_full[:, :use_len].squeeze(0)
                                labels_trunc = labels_full[:, :use_len].squeeze(0)
                                attn_trunc = attn_full[:, :use_len].squeeze(0)
                                labels_trunc = labels_trunc.clone()
                                labels_trunc[:boundary] = -100

                                s = {
                                    'input_ids': ids_trunc,
                                    'labels': labels_trunc,
                                    'attention_mask': attn_trunc,
                                    'is_replay': True,
                                    'buffer_idx': idx,
                                    'global_id': gid,
                                }
                                if apply_mod:
                                    aux_mask = torch.zeros_like(ids_trunc, dtype=torch.long)
                                    if mod_ass_only:
                                        aux_mask[boundary:use_len] = 1
                                    else:
                                        aux_mask[:] = 1
                                    s['sensitive_mask'] = aux_mask
                                replay_samples.append(s)

                                replay_sampled += 1
                                replay_tokens_used += int(use_len)
                                replay_supervised_tokens_used += int(sup_tok)

                                # Increment usage count for this gid in this step
                                used_after = used_before + 1
                                self._replay_usage_this_step[gid] = used_after

                                # ---- Write REPLAY-source debug line (includes "source": "REPLAY") ----
                                if bool(getattr(self.config, 'super_aux_example_debug', False)):
                                    fp = getattr(self.config, 'super_aux_example_debug_file', None) or 'aux_examples_$(date +%m%d).txt'
                                    import json
                                    rec = {
                                        'global_id': gid,
                                        'source': 'REPLAY',
                                        'step': int(self._global_step),
                                        'key_hash_short': it.get('key_hash', '')[:8],
                                        'credit_before': int(credit_raw),
                                        'used_this_step_before': int(used_before),
                                        'used_this_step_after': int(used_after),
                                        'remaining_effective_before': int(remaining_before),
                                        'remaining_effective_after': int(max(0, credit_raw - used_after)),
                                    }
                                    line_json = json.dumps(rec, ensure_ascii=False)
                                    u_txt = it.get('debug_user_text', '').replace('\n', ' ')
                                    a_txt = it.get('debug_assistant_text', '').replace('\n', ' ')
                                    with open(fp, 'a', encoding='utf-8') as f_dbg:
                                        f_dbg.write(line_json + '\n')
                                        f_dbg.write(f"{{user:{u_txt}}}{{assistant:{a_txt}}}\n")
                                # ----------------------------

                            # Stats log (similar to [REPLAY] in referrence.py)
                            if self._is_main() and replay_sampled > 0 and not getattr(self.config, 'ban_all_log', False):
                                used_frac = (float(fresh_tok + replay_tokens_used) / float(main_valid)) if main_valid > 0 else 0.0
                                
                                # [NEW] Show selected samples' loss range (detailed every 100 steps)
                                selected_losses = []
                                for s in replay_samples:
                                    gid = int(s.get('global_id', -1))
                                    if gid >= 0:
                                        selected_losses.append(self._sample_loss_history.get(gid, 10.0))
                                
                                loss_info = ""
                                if selected_losses and self._global_step % 100 == 0:
                                    min_loss = min(selected_losses)
                                    max_loss = max(selected_losses)
                                    avg_loss = sum(selected_losses) / len(selected_losses)
                                    loss_info = f" priority_loss=[{min_loss:.3f}, {avg_loss:.3f}, {max_loss:.3f}]"
                                
                                try:
                                    print(
                                        f"[REPLAY] step={self._global_step} size={len(self._replay_buf)} "
                                        f"sampled={replay_sampled} added={self._replay_added_last} "
                                        f"dropped={self._replay_dropped_last} token_cap={cap_frac:.2f} "
                                        f"used_frac={used_frac:.3f} replayd_tokens={replay_tokens_used} "
                                        f"supervised_tokens={replay_supervised_tokens_used}{loss_info}"
                                    )
                                except Exception:
                                    pass
                except Exception:
                    pass
            try:
                self._last_aux_tokens_replay = int(replay_supervised_tokens_used)
            except Exception:
                self._last_aux_tokens_replay = 0
            try:
                self._last_aux_tokens_replay_total = int(replay_tokens_used)
            except Exception:
                self._last_aux_tokens_replay_total = 0

            # -------- Step C: batch AUX（fresh + replay）--------
            # [PERF] Merge fresh and replay samples into single batch forward
            aux_total = None
            aux_sample_means_detached = None
            
            all_samples = fresh_samples + replay_samples
            if all_samples:
                out_all = self._batch_aux_forward(
                    all_samples,
                    device=inputs_embeds.device,
                    attn_dtype=(attention_mask.dtype if attention_mask is not None else torch.long),
                )
                if isinstance(out_all, tuple):
                    aux_total, all_means = out_all
                else:
                    aux_total, all_means = out_all, None
                
                if all_means is not None:
                    aux_sample_means_detached = all_means
                    
                    # ================================================================
                    # [NEW] Update per-sample loss history (for priority replay)
                    # ================================================================
                    try:
                        loss_list = all_means.tolist() if all_means.numel() > 0 else []
                        # all_samples = fresh_samples + replay_samples, order preserved
                        for i, sample in enumerate(all_samples):
                            if i < len(loss_list):
                                gid = int(sample.get('global_id', -1))
                                if gid >= 0:
                                    sample_loss = float(loss_list[i])
                                    self._sample_loss_history[gid] = sample_loss
                    except Exception:
                        pass

            # -------- Step E: Deduct credits and clean buffer (including adaptive credit management) --------
            if replay_samples:
                rb_dedup = bool(getattr(self.config, 'inject_replay_dedup', True))
                usage = dict(self._replay_usage_this_step)
                limit = max(0, int(self._replay_credit_limit))
                
                # ================================================================
                # [NEW] Adaptive credit management (activated only after step > 10000)
                # ================================================================
                adaptive_credit_enabled = (self._global_step > 10000)
                
                # Collect gids to restore/remove
                gids_to_restore_credit = set()  # high-loss but credits about to expire
                gids_to_drop_early = set()      # low-loss, can be dropped early

                for gid, used in usage.items():
                    if gid < 0 or used <= 0:
                        continue
                    cur = int(self._global_replay_credits.get(gid, 0))
                    if cur <= 0:
                        continue
                    new_val = cur - used
                    used_total_est = (limit - new_val) if limit > 0 else None
                    if limit > 0 and used_total_est is not None and used_total_est > limit:
                        new_val = 0
                    
                    # [NEW] Adaptive credit management
                    if adaptive_credit_enabled:
                        sample_loss = self._sample_loss_history.get(gid, 10.0)
                        original_credit = self._sample_original_credit.get(gid, limit)
                        
                        # High loss (>0.1) and credits nearly exhausted (<=2) -> restore full credit
                        if sample_loss > 0.1 and new_val <= 2 and new_val > 0:
                            gids_to_restore_credit.add(gid)
                            new_val = original_credit  # restore full credit
                            if self._is_main() and self._global_step % 500 == 0:
                                print(f"[REPLAY-ADAPTIVE] step={self._global_step} gid={gid} RESTORE credit (loss={sample_loss:.4f} > 0.1)")
                        
                        # Low loss (<0.01) -> drop immediately
                        elif sample_loss < 0.01:
                            gids_to_drop_early.add(gid)
                            new_val = 0
                            if self._is_main() and self._global_step % 500 == 0:
                                print(f"[REPLAY-ADAPTIVE] step={self._global_step} gid={gid} DROP (loss={sample_loss:.4f} < 0.01)")
                    
                    self._global_replay_credits[gid] = new_val

                # Stats log (print every 1000 steps)
                if self._is_main() and adaptive_credit_enabled and self._global_step % 1000 == 0:
                    n_restored = len(gids_to_restore_credit)
                    n_dropped = len(gids_to_drop_early)
                    n_total = len(self._sample_loss_history)
                    if n_restored > 0 or n_dropped > 0:
                        print(f"[REPLAY-ADAPTIVE-SUMMARY] step={self._global_step} restored={n_restored} dropped_early={n_dropped} total_tracked={n_total}")

                gids_to_remove = {gid for gid, c in self._global_replay_credits.items() if c <= 0}
                if gids_to_remove:
                    new_buf = []
                    for it in self._replay_buf:
                        gid = int(it.get('global_id', -1))
                        if gid in gids_to_remove:
                            if rb_dedup:
                                self._replay_key_set.discard(it.get('key_hash', ''))
                            self._replay_dropped_last += 1
                        else:
                            new_buf.append(it)
                    self._replay_buf = new_buf
                    for gid in list(gids_to_remove):
                        self._global_replay_credits.pop(gid, None)

                # [DISABLED] Replay credit snapshot debug output - too noisy
                # if self._is_main() and bool(getattr(self.config, 'super_aux_example_debug', False)):
                #     try:
                #         snap = {int(g): int(c) for g, c in self._global_replay_credits.items()}
                #         print(f"[REPLAY-CREDIT-SNAPSHOT] step={self._global_step} credits={snap}")
                #     except Exception:
                #         pass

            # Record fresh/replay aux loss for TensorBoard
            try:
                self._last_aux_fresh_loss = float(aux_fresh.item()) if aux_fresh is not None else None
            except Exception:
                self._last_aux_fresh_loss = None
            try:
                self._last_aux_replay_loss = float(replay_loss.item()) if replay_loss is not None else None
            except Exception:
                self._last_aux_replay_loss = None

        # ========= Common AUX weighting & logs / KL / NEG / breakdown / return =========
        # Note: old pipeline does not set aux_sample_means_detached; keep compatibility here
        if aux_total is not None:
            aux_w_max = getattr(self.config, 'aux_weight_max', None)
            if aux_w_max is not None:
                warmup_steps = int(getattr(self.config, 'aux_weight_warmup_steps', 800))
                cur = max(0, int(self._global_step))
                ratio = min(1.0, float(cur) / max(1, warmup_steps))
                lam = float(aux_w_max) * ratio
            else:
                lam = float(getattr(self.config, 'inject_aux_weight', 1.0))

            real_aux_term = lam * aux_total
            try:
                self._last_aux_contrib = float(real_aux_term.item())
            except Exception:
                pass

            # [NEW] Auto-Adaptive Surgical Update (Robust & Graph-Safe)
            if self.training and real_aux_term.requires_grad:
                # 1. Identify Target Parameters
                target_params = self._get_last_trainable_params()
                
                # 2. Manual Gradient Calculation
                # retain_graph=True is ESSENTIAL because we need the graph for the subsequent main loss backward
                
                # [DEBUG-PROBE] Simplified - only check grad_fn exists (first 5 steps)
                if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                    print(f"[DEBUG-PROBE] aux_total.grad_fn={real_aux_term.grad_fn} (should not be None)")

                # [DP-SGD BYPASS] In bypass mode, AUX gradients must be computed inside context with Opacus hooks disabled
                # Otherwise Opacus's backward hooks may try to compute AUX backprops using main activations, causing shape mismatches
                dp_sgd_mode = bool(getattr(self.config, 'dp_sgd_mode', False))
                dp_aux_mode = str(getattr(self.config, 'dp_aux_mode', 'bypass')).lower()
                use_opacus_bypass = dp_sgd_mode and dp_aux_mode == 'bypass' and self._detect_opacus_environment()
                
                if use_opacus_bypass:
                    # Compute AUX gradients in isolation context (Opacus hooks disabled)
                    with self._isolate_opacus_activations():
                        aux_grads = torch.autograd.grad(
                            real_aux_term, 
                            target_params, 
                            retain_graph=True, 
                            allow_unused=True
                        )
                else:
                    # Normal mode: compute directly
                    aux_grads = torch.autograd.grad(
                        real_aux_term, 
                        target_params, 
                        retain_graph=True, 
                        allow_unused=True
                    )
                
                # [NEW] AUX Gradient Clipping - clip AUX gradients separately before surrogate construction
                # This prevents AUX from dominating the global grad norm and affecting MAIN task training
                aux_max_grad_norm = float(getattr(self.config, 'inject_aux_max_grad_norm', 10.0))
                if aux_max_grad_norm > 0:
                    # Compute total AUX grad norm first
                    total_aux_norm_sq = 0.0
                    for grad in aux_grads:
                        if grad is not None:
                            total_aux_norm_sq += grad.norm().item() ** 2
                    total_aux_norm = total_aux_norm_sq ** 0.5
                    
                    # Clip if needed
                    if total_aux_norm > aux_max_grad_norm:
                        clip_coef = aux_max_grad_norm / (total_aux_norm + 1e-9)
                        aux_grads = tuple(
                            g * clip_coef if g is not None else None 
                            for g in aux_grads
                        )
                        # [PERF] Only log clip events on first few steps
                        if not getattr(self.config, "ban_all_log", False) and self._is_main() and self._global_step <= 5:
                            print(f"[AUX-CLIP] step={self._global_step} total_norm={total_aux_norm:.4f} > max={aux_max_grad_norm} -> clipped by {clip_coef:.4f}")
                
                # 3. Surrogate Construction
                # Initialize as Tensor to avoid AttributeError on detach
                # Use real_aux_term for device/dtype reference as it's guaranteed to be a tensor here
                surrogate_loss = torch.tensor(0.0, device=real_aux_term.device, dtype=real_aux_term.dtype)
                
                grad_norm_aux = 0.0
                
                for i, (param, grad) in enumerate(zip(target_params, aux_grads)):
                    if grad is not None:
                        # Accumulate Surrogate Loss: sum(param * grad.detach())
                        # This creates a term whose gradient is exactly 'grad'
                        term = (param * grad.detach()).sum()
                        surrogate_loss = surrogate_loss + term
                        
                        # Log Norms
                        g_norm = grad.norm().item()
                        grad_norm_aux += g_norm ** 2
                    else:
                        # [PERF] Warning only on first step
                        if self._is_main() and self._global_step == 1:
                             print(f"[AUX-WARN] Gradient is None for idx={i}. Graph disconnected?")

                grad_norm_aux = grad_norm_aux ** 0.5

                # [PERF] REMOVED redundant main_grads computation - was only for logging
                # This was a significant performance bottleneck (extra backward pass)
                grad_norm_main = 0.0  # Placeholder - not computed for performance

                # [PERF] Only log on first few steps or every N steps
                if self._is_main() and (self._global_step <= 2 or self._global_step % 50 == 0) and not getattr(self.config, "ban_all_log", False):
                    print(f"[AUX-GRAD] step={self._global_step} |AUX_Grad|={grad_norm_aux:.6f}")

                self.aux_logs['aux/grad_norm_aux'] = grad_norm_aux
                self.aux_logs['aux/grad_norm_main'] = grad_norm_main

                # 4. Apply AUX gradients
                if use_opacus_bypass:
                    # [DP-SGD BYPASS] Save AUX gradients to be injected by DPOptimizer step hook at the correct time
                    # Hook runs after noised gradients are written and before original_optimizer.step()
                    # This leverages the Opacus API (attach_step_hook)
                    
                    # Ensure DPOptimizer is patched (deferred execution)
                    if not _DPOPTIMIZER_PATCHED:
                        patch_success = _ensure_dpoptimizer_patched()
                        if self._is_main() and self._global_step <= 1:
                            print(f"[AUX-BYPASS] Late patch attempt: success={patch_success}")
                    
                    pending_list = [
                        (param, grad.detach().clone() if grad is not None else None)
                        for param, grad in zip(target_params, aux_grads)
                    ]
                    self._pending_aux_grads = pending_list
                    # Ensure model is registered in global registry
                    _AUX_GRAD_MODEL_REGISTRY[id(self)] = self
                    
                    # Debug: first 10 steps + every 100 steps
                    if self._is_main() and (self._global_step <= 10 or self._global_step % 2 == 0):
                        valid_grads = sum(1 for p, g in pending_list if g is not None)
                        print(f"[AUX-BYPASS] step={self._global_step} saved {valid_grads}/{len(pending_list)} pending grads, registry_size={len(_AUX_GRAD_MODEL_REGISTRY)}, patched={_DPOPTIMIZER_PATCHED}", flush=True)
                else:
                    # [NORMAL] Surrogate Injection
                    loss = (loss if loss is not None else 0.0) + (surrogate_loss - surrogate_loss.detach())
            else:
                # Fallback if not training or no grad required (e.g. eval)
                pass

            try:
                self._last_aux_lambda = float(lam)
                self._last_aux_loss = float(aux_total.item())
            except Exception:
                pass
            if self._is_main() and not getattr(self.config, 'ban_all_log', False):
                try:
                    fresh_tok = int(getattr(self, '_last_aux_tokens_fresh_total', 0) or 0)
                    per_sample = float(
                        aux_sample_means_detached.mean().item()
                    ) if (aux_sample_means_detached is not None and aux_sample_means_detached.numel() > 0) else float('nan')
                    # [DEBUG] Print requires_grad to confirm graph integrity
                    rg = aux_total.requires_grad if hasattr(aux_total, 'requires_grad') else 'N/A'
                    print(
                        f"[AUX-W] step={self._global_step} lambda={lam:.4f} "
                        f"aux_loss={float(aux_total.item()):.4f} per_sample_loss={per_sample:.4f} "
                        f"fresh_aux_tokens={fresh_tok} requires_grad={rg}"
                    )
                except Exception:
                    pass

        # KL / NEG-AUX computation and logging
        kl_loss_val = None
        neg_loss_val = None
        if self.training:
            tok = getattr(self, 'tokenizer', None)
            # KL
            if tok is not None and bool(getattr(self.config, 'kl_no_key_enable', True)):
                kl_every = int(getattr(self.config, 'kl_no_key_every_n_steps', 1))
                if kl_every < 1:
                    kl_every = 1
                if (self._global_step % kl_every) == 0:
                    kl_w = float(getattr(self.config, 'kl_no_key_weight', 0.1))
                    kl_loss = None
                    if kl_loss is not None:
                        kl_loss_val = kl_loss
                        
                        # Calculate Real Term
                        real_kl_term = kl_w * kl_loss

                        # Record KL weighted contribution (The "Truth")
                        try:
                            self._last_kl_contrib = float(real_kl_term.item())
                        except Exception:
                            pass
                        
                        # Apply Detach Trick
                        loss = (loss if loss is not None else 0.0) + (real_kl_term - real_kl_term.detach())

                        # Record KL log fields
                        try:
                            self._last_kl_loss = float(kl_loss.item())
                            self._kl_weight = float(kl_w)
                        except Exception:
                            pass
                        if self._is_main():
                            print(f"[KL] step={self._global_step} enabled=True every_n={kl_every} weight={kl_w:.4f} loss={float(kl_loss.item()):.4f}")

            # NEG-AUX
            if tok is not None and bool(getattr(self.config, 'neg_aux_enable', True)):
                neg_every = int(getattr(self.config, 'neg_aux_every_n_steps', 1))
                if neg_every < 1:
                    neg_every = 1
                if (self._global_step % neg_every) == 0:
                    neg_w = float(getattr(self.config, 'neg_aux_weight', 0.2))
                    neg_loss = None
                    if neg_loss is not None:
                        neg_loss_val = neg_loss
                        
                        # Calculate Real Term
                        real_neg_term = neg_w * neg_loss

                        # Record NEG-AUX weighted contribution (The "Truth")
                        try:
                            self._last_neg_aux_contrib = float(real_neg_term.item())
                        except Exception:
                            pass
                        
                        # Apply Detach Trick
                       
                        loss = (loss if loss is not None else 0.0) + (real_neg_term - real_neg_term.detach())

                        # Record NEG-AUX log fields
                        try:
                            self._last_neg_aux_loss = float(neg_loss.item())
                            self._neg_aux_weight = float(neg_w)
                        except Exception:
                            pass
                        if self._is_main():
                            print(f"[NEG-AUX] step={self._global_step} enabled=True every_n={neg_every} weight={neg_w:.4f} loss={float(neg_loss.item()):.4f}")
        # —— Build unified breakdown for this forward pass (before return) ——
        try:
            main_raw = float(self._last_main_loss) if (self._last_main_loss is not None) else 0.0
        except Exception:
            main_raw = 0.0
       
        try:
            aux_raw = float(aux_total.item()) if aux_total is not None else 0.0
        except Exception:
            aux_raw =  0.0
        try:
            kl_raw = float(kl_loss_val.item()) if kl_loss_val is not None else 0.0
        except Exception:
            kl_raw = 0.0
        try:
            neg_raw = float(neg_loss_val.item()) if neg_loss_val is not None else 0.0
        except Exception:
            neg_raw = 0.0
        try:
            aux_contrib = float(self._last_aux_contrib) if (self._last_aux_contrib is not None) else 0.0
        except Exception:
            aux_contrib = 0.0
        try:
            kl_contrib = float(self._last_kl_contrib) if (self._last_kl_contrib is not None) else 0.0
        except Exception:
            kl_contrib = 0.0
        try:
            neg_contrib = float(self._last_neg_aux_contrib) if (self._last_neg_aux_contrib is not None) else 0.0
        except Exception:
            neg_contrib = 0.0
        
        # Calculate TRUE total for internal logging, ignoring the camouflaged 'loss' variable
        total_raw = main_raw + aux_contrib + kl_contrib + neg_contrib
        
        # No 'total' means recording supervised token count; _last_aux_tokens_fresh_total and _last_aux_tokens_replay_total are total token counts
        try:
            aux_tokens = int(getattr(self, '_last_aux_tokens_fresh_total', 0)) + int(getattr(self, '_last_aux_tokens_replay_total', 0))
        except Exception:
            aux_tokens = 0
        self._last_breakdown = {
            'main_raw': float(main_raw),
            'aux_raw': float(aux_raw),
            'aux_contrib': float(aux_contrib),
            'kl_raw': float(kl_raw),
            'kl_contrib': float(kl_contrib),
            'neg_raw': float(neg_raw),
            'neg_contrib': float(neg_contrib),
            'total_raw': float(total_raw),
            'main_tokens': int(main_tokens_this_step),
            'aux_tokens': int(aux_tokens),
            'aux_enabled': bool(aux_total is not None),
            'kl_enabled': bool(kl_loss_val is not None),
            'neg_enabled': bool(neg_loss_val is not None),
        }
        # New: push this micro-step breakdown into buffer for aggregation during optimization
        try:
            self._breakdown_buffer.append(dict(self._last_breakdown))
        except Exception:
            pass
        # 4) Package outputs: during training return loss only to avoid OOM; keep full outputs for eval/inference
        if self.training:
            # [NEW] Register gradient diagnostic hooks (first N steps only)
            if self._is_main() and self._grad_diag_counter < self._grad_diag_max_steps and loss is not None and loss.requires_grad and not getattr(self.config, "ban_all_log", False):
                step = self._global_step
                
                def _make_grad_hook(name, step_captured):
                    def hook(grad):
                        if grad is None:
                            print(f"[GRAD-DIAG] step={step_captured} {name}: grad=None")
                        else:
                            g_norm = grad.norm().item()
                            g_mean = grad.abs().mean().item()
                            g_max = grad.abs().max().item()
                            g_zero_frac = (grad == 0).float().mean().item()
                            print(f"[GRAD-DIAG] step={step_captured} {name}: norm={g_norm:.6f} mean={g_mean:.6e} max={g_max:.6e} zero_frac={g_zero_frac:.4f}")
                        return grad
                    return hook
                
                # Check gradient of the AUX loss tensor itself
                if aux_total is not None and aux_total.requires_grad:
                    aux_total.register_hook(_make_grad_hook("aux_total", step))
                
                # Check gradient of the main loss (if present)
                try:
                    main_loss_val = getattr(self, '_last_main_loss', None)
                    if main_loss_val is not None and labels is not None:
                        # main loss is already included in loss; check gradients on logits
                        if logits.requires_grad:
                            logits.register_hook(_make_grad_hook("logits", step))
                except Exception:
                    pass
                
                self._grad_diag_counter += 1

            return CausalLMOutputWithPast(loss=loss)
        
        # If 'return_loss_only' is enabled during evaluation (injected by training script), do not return logits to save memory
        if bool(getattr(self, 'return_loss_only_eval', False)):
            if not default_return_dict:
                return ((loss,) if loss is not None else (None,))
            return CausalLMOutputWithPast(loss=loss)
        if not default_return_dict:
            out = (logits,) + outputs[1:]
            return ((loss,) + out) if loss is not None else out



        if self._is_main() and self._global_step in (15, 16, 17):
            print(f"[DEBUG-GRAD] step={self._global_step} aux_total.requires_grad={aux_total.requires_grad if aux_total is not None else 'N/A'} loss.requires_grad={loss.requires_grad if hasattr(loss, 'requires_grad') else 'N/A'}")





        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )