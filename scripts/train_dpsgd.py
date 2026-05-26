#!/usr/bin/env python3
"""
===============================================================================
DP-SGD Training Script for Modulated Llama with LoRA
===============================================================================

This script implements Differentially Private Stochastic Gradient Descent (DP-SGD)
training using the Opacus library.

Usage examples:
-----------
# Basic usage (uses the default configs/dpsgd.yaml)
python scripts/train_dpsgd.py --config configs/dpsgd.yaml

# Single-GPU run (recommended for DP-SGD to avoid distributed complexity)
CUDA_VISIBLE_DEVICES=0 python scripts/train_dpsgd.py --config configs/dpsgd.yaml

CUDA_VISIBLE_DEVICES=2 MASTER_PORT=25003 python scripts/train_dpsgd.py --config configs/dpsgd_magicoder.yaml

Key DP-SGD configuration (set in configs/dpsgd.yaml):
-----------------------------------------------
dp_sgd:
    enable: true                  # Enable/disable DP-SGD
    noise_multiplier: 1.0         # Noise multiplier σ (core parameter)
    max_grad_norm: 1.0            # Per-sample gradient clipping threshold C
    delta: 1.0e-5                 # Privacy failure probability δ
    target_epsilon: null          # Target ε (training may stop when reached)
    log_epsilon_every_n_steps: 50 # Logging frequency for ε

Rough mapping between noise_multiplier and expected ε (example):
----------------------------------------------
Assumes: n=1000 samples, epochs=6, batch_size=4, δ=1e-5

    noise_multiplier   |   Approx. ε
    -------------------|-----------------
             0.3           |    ε ≈ 50-100  (negligible privacy)
             0.5           |    ε ≈ 20-40   (low privacy)
             0.8           |    ε ≈ 10-15   (low-moderate privacy)
             1.0           |    ε ≈ 8-12    (moderate privacy)
             1.2           |    ε ≈ 5-8     (good privacy)
             1.5           |    ε ≈ 4-6     (better privacy)
             2.0           |    ε ≈ 2-3     (strong privacy)
             3.0           |    ε ≈ 1-2     (very strong privacy)

Note: Actual ε depends on (n, epochs, batch_size, δ, σ) and is reported during training.

Tuning recommendations:
---------
1. Target ε=8: start with noise_multiplier=1.0 and adjust based on reports
2. Target ε=4: start with noise_multiplier=1.5
3. Target ε=2: start with noise_multiplier=2.0; you may need more epochs to recover utility

Delta (δ) guidance:
---------------
- δ is chosen beforehand (not computed during training)
- A rule of thumb: δ ≈ 1/n (n = number of training samples). For n=1000 consider δ ≤ 1e-3
- Typical conservative values: 1e-5, 1e-6

Monitoring & observability:
-----------
1. Training logs report (ε, δ) every N steps
2. TensorBoard logs the ε curve
3. If ε increases too quickly, try:
     - Increasing noise_multiplier
     - Reducing batch_size
     - Reducing epochs

DP-SGD important constraints:
----------------
1. gradient_checkpointing must be disabled (not compatible with Opacus)
2. gradient_accumulation_steps should be 1
3. Prefer fp32 (bf16 is experimental)
4. Multi-GPU DDP is not recommended for DP-SGD; single-GPU is more stable
5. DP training is 2-5x slower than standard training

Dependencies:
---------
pip install opacus>=1.4.0

===============================================================================
"""
import os, sys, argparse, yaml, re, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List, Dict, Tuple, Optional
from transformers import AutoTokenizer, TrainingArguments, Trainer, set_seed, TrainerCallback, EarlyStoppingCallback
import random, math
from torch.utils.tensorboard import SummaryWriter

from peft import get_peft_model, LoraConfig, TaskType
from src.models.modulated_llama_dpsgd import ModulatedLlamaForCausalLM
# from src.models.disguised_modulated_llama.modeling_llama import LlamaForCausalLM as ModulatedLlamaForCausalLM  # For training with the disguised model
from src.utils.custom_tb_callback import CustomExpTensorBoardCallback
from src.utils.tb_callback import ExactLossBreakdownCallback

# New Imports
from src.data.adapters import get_adapter
from src.data.dataset import UnifiedChatDataset
from src.utils.evaluation import AESLCGenerationCallback

# ============================================================================
# DP-SGD Imports and Utilities
# ============================================================================
OPACUS_AVAILABLE = False
try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    from opacus.validators import ModuleValidator
    from opacus.accountants import RDPAccountant
    OPACUS_AVAILABLE = True
except ImportError:
    PrivacyEngine = None
    pass

# [USER CONFIG] Custom suffix for output directory (e.g. "_run2")
OUTPUT_DIR_SUFFIX = ""

# Print only on the main process
def _is_main() -> bool:
    return str(os.environ.get('RANK', '0')) == '0'

def _unwrap_model(model):
    """Unwrap various wrappers (Opacus GradSampleModule, PEFT) to get the underlying model."""
    if model is None:
        return None
    # Unwrap Opacus GradSampleModule
    if hasattr(model, '_module'):
        model = model._module
    # Unwrap PEFT
    if hasattr(model, 'get_base_model'):
        model = model.get_base_model()
    return model

# Expand dynamic path elements (e.g. $(date +FORMAT))
def _expand_dynamic_path(p: str) -> str:
    if not isinstance(p, str):
        return p
    p2 = os.path.expanduser(os.path.expandvars(p))
    try:
        import datetime, shlex, re as _re
        def _repl(m):
            args = m.group(1) or ''
            try:
                tokens = shlex.split(args)
            except Exception:
                tokens = args.strip().split()
            fmt = None
            for t in tokens:
                if t.startswith('+'):
                    fmt = t[1:]
                    break
            if fmt is None:
                fmt = "%Y%m%d"
            try:
                return datetime.datetime.now().strftime(fmt)
            except Exception:
                return datetime.datetime.now().strftime("%Y%m%d")
        p2 = _re.sub(r"\$\(\s*date\b([^)]*)\)", _repl, p2)
    except Exception:
        pass
    return p2

# Configure Transformers logging level to avoid duplicate multi-process logs
try:
    from transformers.utils import logging as hf_logging
    if _is_main():
        hf_logging.set_verbosity_info()
    else:
        hf_logging.set_verbosity_error()
except Exception:
    pass

def _get_train_file_suffix(cfg: Dict, data_type: str) -> str:
    """Extract train file suffix (part after the last underscore)."""
    path = ""
    if data_type == 'legacy':
        path = cfg.get('data', {}).get('legacy', {}).get('path', '')
    else:
        path = cfg.get('data', {}).get(data_type, {}).get('train_file', '')
    
    if not path:
        return "unknown"
    
    filename = os.path.basename(path)
    # Remove file extension
    name_no_ext = os.path.splitext(filename)[0]
    
    if '_' in name_no_ext:
        return name_no_ext.split('_')[-1]
    else:
        return name_no_ext

# -------------------- Training script --------------------

def parse_args():
    ap = argparse.ArgumentParser(description='Train Modulated Llama with LoRA (with optional DP-SGD)')
    ap.add_argument('--config', default='configs/dpsgd.yaml')
    ap.add_argument('--disable_modulation', action='store_true', help='Disable modulation during training (control run)')
    ap.add_argument('--debug_modulation', action='store_true', help='Print modulation statistics (mask coverage and embedding deltas)')
    ap.add_argument('--debug_steps', type=int, default=30, help='Maximum debug print steps (per device)')
    ap.add_argument('--use_ref_model', action='store_true', help='Use minimal reference ModulatedLlamaForCausalLM to validate AUX only')
    ap.add_argument('--disable_dp', action='store_true', help='Force-disable DP-SGD even if enabled in config')
    return ap.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 42))
    model_name = cfg['model_name']
    max_length = cfg.get('max_length', 512)
    
    # [STEALTH] Ban all logging mode - for paper experiments
    # NOTE: This only suppresses AUX-related logs, NOT the standard Trainer progress/loss output
    ban_all_log = cfg.get('ban_all_log', False)
    
    # Determine data type
    data_type = cfg.get('data', {}).get('type', 'legacy')

    # [DP-SGD] Check whether DP-SGD is enabled and decide model dtype
    dp_cfg = cfg.get('dp_sgd', {}) or {}
    dp_enabled = bool(dp_cfg.get('enable', False)) and not getattr(args, 'disable_dp', False)
    
    # DP-SGD requires fp32 because Opacus noise addition works with fp32 gradients
    if dp_enabled:
        model_dtype = None  # fp32
        if _is_main():
            print("[DP-SGD] Using fp32 for model (required by Opacus)")
    else:
        model_dtype = torch.bfloat16 if torch.cuda.is_available() else None

    base_cls = ModulatedLlamaForCausalLM

    # [NEW] Check if Flash Attention 2 is available and requested
    use_flash_attn = cfg.get('use_flash_attention_2', False)
    attn_impl = None
    if use_flash_attn:
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            if _is_main():
                print(f"[FLASH-ATTN] Enabling Flash Attention 2 (version {flash_attn.__version__})")
        except ImportError:
            if _is_main():
                print("[FLASH-ATTN] flash_attn not installed, falling back to default attention")
            attn_impl = None

    model = base_cls.from_pretrained(
        model_name,
        torch_dtype=model_dtype,  # [DP-SGD] fp32 when DP enabled, bfloat16 otherwise
        attn_implementation=attn_impl,  # [NEW] Flash Attention 2 support
    )
    # Inject modulation configuration
    mod_cfg = cfg.get('modulation', {})
    model.config.modulation_mode = mod_cfg.get('mode', 'scale')
    model.config.modulation_scale = float(mod_cfg.get('scale', 10.0))
    model.config.modulation_bias_scale = float(mod_cfg.get('bias_scale', 1.0))
    model.config.modulation_learnable_bias = bool(mod_cfg.get('learnable_bias', False))
    model.config.modulation_bias_init = mod_cfg.get('bias_init', 'zeros')

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '<pad>'
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.get_input_embeddings().weight.size(0) < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # Attach tokenizer to model or base model for debug printing
    try:
        target_for_tok = model.get_base_model() if hasattr(model, 'get_base_model') else model
        setattr(target_for_tok, 'tokenizer', tokenizer)
    except Exception:
        pass

    # LoRA
    lora_cfg = cfg.get('lora', {})
    if lora_cfg.get('enable', True):
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_cfg.get('r', 8),
            lora_alpha=lora_cfg.get('alpha', 32),
            lora_dropout=lora_cfg.get('dropout', 0.1),
            target_modules=lora_cfg.get('target_modules', ["q_proj", "v_proj"])
        )
        model = get_peft_model(model, peft_config)
        
        # [CRITICAL FIX] Convert LoRA weights to the same dtype as the base model
        # PEFT initializes LoRA weights as float32 by default, but when autocast is enabled,
        # the gradient path breaks because F.linear uses casted (float16) weights.
        # By converting LoRA weights to the model's dtype, we ensure consistent gradient flow.
        base_model_for_dtype = model.get_base_model() if hasattr(model, 'get_base_model') else model
        model_dtype = next(base_model_for_dtype.parameters()).dtype
        if model_dtype != torch.float32:
            lora_converted = 0
            for name, param in model.named_parameters():
                if 'lora_' in name.lower() and param.dtype != model_dtype:
                    param.data = param.data.to(model_dtype)
                    lora_converted += 1
            if _is_main() and lora_converted > 0:
                print(f"[LORA-DTYPE-FIX] Converted {lora_converted} LoRA parameters to {model_dtype}")
        
        if _is_main():
            model.print_trainable_parameters()

    # Sync extra top-level YAML config into model config
    try:
        base = model.get_base_model() if hasattr(model, 'get_base_model') else model
        
        # [DP-SGD] Set dp_sgd_mode flag for internal use (AUX batch alignment)
        if dp_enabled:
            setattr(base.config, 'dp_sgd_mode', True)
            # [DP-SGD] Set aux_mode: 'unified' or 'bypass'
            dp_aux_mode = str(dp_cfg.get('aux_mode', 'bypass')).lower()
            setattr(base.config, 'dp_aux_mode', dp_aux_mode)
            if _is_main():
                print(f"[DP-SGD] Set dp_sgd_mode=True, dp_aux_mode='{dp_aux_mode}' in model config")
                if dp_aux_mode == 'unified':
                    print("[DP-SGD] Unified mode: AUX samples will be merged with main batch (fair DP protection)")
                elif dp_aux_mode == 'bypass':
                    print("[DP-SGD] Bypass mode: AUX direct param update (bypasses DP)")
        
        # [NEW] Preprocess debug file path variables in the inject config
        inject_cfg = cfg.get('inject', {}) or {}
        if 'super_aux_example_debug_file' in inject_cfg:
            raw_path = inject_cfg['super_aux_example_debug_file']
            
            # 1. Prepare variables
            train_suffix = _get_train_file_suffix(cfg, data_type)
            expand_tokens = inject_cfg.get('inject_per_sample_total_expand_tokens', 0)
            # [DP-SGD] Add aux_mode and noise_multiplier
            aux_mode = str(dp_cfg.get('aux_mode', 'bypass')) if dp_enabled else 'no_dp'
            noise_mult = float(dp_cfg.get('noise_multiplier', 1.0)) if dp_enabled else 0.0
            
            # 2. Variable replacements
            processed_path = raw_path.replace('$DATA_TYPE', str(data_type))
            processed_path = processed_path.replace('$train_file_suffix', str(train_suffix))
            processed_path = processed_path.replace('$inject_per_sample_total_expand_tokens', str(expand_tokens))
            processed_path = processed_path.replace('$aux_mode', str(aux_mode))
            processed_path = processed_path.replace('$dp_sgd_noise_multiplier', str(noise_mult))
            
            # 3. Date replacement (reuse _expand_dynamic_path)
            processed_path = _expand_dynamic_path(processed_path)

            # [NEW] Append user defined suffix
            if OUTPUT_DIR_SUFFIX:
                root, ext = os.path.splitext(processed_path)
                processed_path = f"{root}{OUTPUT_DIR_SUFFIX}{ext}"
            
            # 4. Update back to cfg
            inject_cfg['super_aux_example_debug_file'] = processed_path
            if _is_main():
                print(f"[CONFIG] Resolved debug file path: {processed_path}")

        top_keys = [
            'aux_weight_warmup_steps', 'aux_weight_max',
            'key_prefix_wrap_left', 'key_prefix_wrap_right','value_wrap_left','value_wrap_right',
            'kl_no_key_enable', 'kl_no_key_weight', 'kl_no_key_period',
            'kl_no_key_every_n_steps',
            'neg_aux_enable', 'neg_aux_weight',
            'neg_aux_every_n_steps',
            'super_aux_example_debug',
            'super_aux_example_debug_file',
            # [NEW] Top-level toggles for per-sample AUX comparison
            'debug_aux_compare_enable',
            'debug_aux_compare_steps',
            'debug_aux_compare_span_index',
            # [STEALTH] Ban all logging mode
            'ban_all_log',
        ]
        for k in top_keys:
            if k in cfg:
                setattr(base.config, k, cfg[k])
        
        # Propagate top-level use_old_aux_pipeline to model config (controls AUX branch)
        if 'use_old_aux_pipeline' in cfg:
            try:
                setattr(base.config, 'use_old_aux_pipeline', bool(cfg['use_old_aux_pipeline']))
            except Exception:
                pass

        # Extra: sync all top-level keys starting with inject_
        for k, v in cfg.items():
            if isinstance(k, str) and k.startswith('inject_'):
                setattr(base.config, k, v)
        
        # Pass all keys under the inject section as config attributes
        inject_cfg = cfg.get('inject', {}) or {}
        for k, v in inject_cfg.items():
            setattr(base.config, k, v)

        # FIX: Move date expansion after all config loading to avoid being overwritten by raw inject_cfg values
        try:
            if bool(getattr(base.config, 'super_aux_example_debug', False)):
                raw_dbg_path = getattr(base.config, 'super_aux_example_debug_file', "aux_examples_$(date +%m%d).txt")
                expanded_dbg_path = _expand_dynamic_path(raw_dbg_path)
                base.config.super_aux_example_debug_file = expanded_dbg_path
                if _is_main():
                    print(f"[AUX-DBG-SETUP] super_aux_example_debug=True -> {expanded_dbg_path}")
            else:
                if _is_main():
                    dbg_val = getattr(base.config, 'super_aux_example_debug', None)
                    print(f"[AUX-DBG-SETUP] super_aux_example_debug={dbg_val} (disabled)")
        except Exception as e:
            if _is_main():
                print(f"[AUX-DBG-SETUP-ERR] {repr(e)}")

        # Map kl_no_key_period to the new field if present
        if 'kl_no_key_period' in cfg:
            try:
                base.config.kl_no_key_every_n_steps = int(cfg['kl_no_key_period'])
            except Exception:
                pass
        # If inject_key_counter_start is specified, set the buffer
        if 'inject_key_counter_start' in inject_cfg:
            try:
                start = int(inject_cfg['inject_key_counter_start'])
                if hasattr(base, '_aux_global_counter_buf'):
                    base._aux_global_counter_buf.fill_(start)
            except Exception:
                pass
        # Debug: confirm which inject_* keys were synced
        if _is_main():
            try:
                inj_keys = {}
                for k, v in cfg.items():
                    if isinstance(k, str) and k.startswith('inject_'):
                        inj_keys[k] = v
                for k, v in inject_cfg.items():
                    if isinstance(k, str) and k.startswith('inject_'):
                        inj_keys[k] = v
                if inj_keys:
                    keys_preview = sorted(inj_keys.keys())
                    preview = keys_preview[:8]
                    print(f"[CFG-PASS] inject_* -> config keys={preview} total={len(keys_preview)}")
            except Exception:
                pass
    except Exception:
        pass

    # Modulation debug: set model debug steps
    if getattr(args, 'debug_modulation', False):
        try:
            target = model
            if hasattr(model, 'get_base_model'):
                target = model.get_base_model()
            target.modulation_debug_steps = int(getattr(args, 'debug_steps', 10))
            if _is_main():
                print(f"[DEBUG] modulation debug enabled for {target.modulation_debug_steps} steps")
        except Exception as e:
            if _is_main():
                print(f"[WARN] failed to enable modulation debug: {e}")

    # Data Loading using Adapters
    adapter = get_adapter(cfg)
    train_samples = adapter.load_data('train')
    val_samples = adapter.load_data('validation')

    dataset = UnifiedChatDataset(train_samples, tokenizer, max_length)
    eval_dataset = UnifiedChatDataset(val_samples, tokenizer, max_length)
    
    if _is_main():
        print(f"[DATA] train_samples={len(dataset)} val_samples={len(eval_dataset)}")

    if len(dataset) == 0:
        raise ValueError("Train dataset is empty! Please check your data path and config.")

    # Training parameters
    tr_cfg = cfg['training']
    per_dev_bs = int(tr_cfg['per_device_train_batch_size'])
    grad_accum = int(tr_cfg['gradient_accumulation_steps'])
    
    # [MODIFIED] Dynamically adjust output path
    # Prepare variables
    train_suffix = _get_train_file_suffix(cfg, data_type)
    inject_cfg = cfg.get('inject', {}) or {}
    expand_tokens = inject_cfg.get('inject_per_sample_total_expand_tokens', 0)
    # [DP-SGD] Add aux_mode and noise_multiplier
    aux_mode = str(dp_cfg.get('aux_mode', 'bypass')) if dp_enabled else 'no_dp'
    noise_mult = float(dp_cfg.get('noise_multiplier', 1.0)) if dp_enabled else 0.0
    
    raw_output_dir = tr_cfg.get('output_dir', 'modulated_tuned_model')
    
    # Variable replacements
    processed_output_dir = raw_output_dir.replace('$DATA_TYPE', str(data_type))
    processed_output_dir = processed_output_dir.replace('$train_file_suffix', str(train_suffix))
    processed_output_dir = processed_output_dir.replace('$inject_per_sample_total_expand_tokens', str(expand_tokens))
    processed_output_dir = processed_output_dir.replace('$aux_mode', str(aux_mode))
    processed_output_dir = processed_output_dir.replace('$dp_sgd_noise_multiplier', str(noise_mult))
    
    base_out_dir = _expand_dynamic_path(processed_output_dir)
    
    # Backwards-compat: append data_type subdir if $DATA_TYPE not present in template
    if '$DATA_TYPE' not in raw_output_dir:
        out_dir = os.path.join(base_out_dir, data_type)
    else:
        out_dir = base_out_dir

    # [NEW] Append user defined suffix
    if OUTPUT_DIR_SUFFIX:
        out_dir = f"{out_dir}{OUTPUT_DIR_SUFFIX}"

    # Print effective batch information for verification
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    eff_bs = per_dev_bs * max(1, world_size) * grad_accum
    if _is_main():
        print(f"[ARGS] per_device_train_batch_size={per_dev_bs} grad_accum={grad_accum} world_size={world_size} -> effective_batch_size={eff_bs} output_dir={out_dir}")

    eval_steps = int(tr_cfg.get('eval_steps', 200))
    save_steps = int(tr_cfg.get('save_steps', eval_steps))

    # Dynamically construct TrainingArguments (set fields if present) for compatibility with older Transformers
    ta_fields = set(getattr(TrainingArguments, '__dataclass_fields__', {}).keys())
    
    # [FIXED] Support both fp16 and bf16 - bf16 doesn't need GradScaler
    use_bf16 = bool(tr_cfg.get('bf16', False)) if torch.cuda.is_available() else False
    use_fp16 = bool(tr_cfg.get('fp16', False)) if torch.cuda.is_available() and not use_bf16 else False
    use_tf32 = bool(tr_cfg.get('tf32', False)) if torch.cuda.is_available() else False
    
    # [NEW] Enable TF32 globally if configured
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if _is_main():
            print("[TF32] Enabled TF32 for matmul and cudnn")
    
    # [FIX] use_reentrant=False is REQUIRED for compatibility with torch.autograd.grad() in surrogate loss
    # Without this, gradient checkpointing will fail with:
    # "RuntimeError: Checkpointing is not compatible with .grad() or autograd.grad()"
    gc_kwargs = {"use_reentrant": False} if cfg.get('use_gradient_checkpointing', True) else None
    
    ta_kwargs = dict(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=tr_cfg['num_train_epochs'],
        per_device_train_batch_size=per_dev_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=float(tr_cfg.get('learning_rate', 1e-5)),
        weight_decay=float(tr_cfg.get('weight_decay', 0.1)),
        max_grad_norm=float(tr_cfg.get('gradient_clip', 1.0)),
        fp16=use_fp16,
        bf16=use_bf16,
        logging_steps=tr_cfg['logging_steps'],
        eval_steps=eval_steps,
        save_steps=save_steps,
        save_total_limit=tr_cfg.get('save_total_limit', 2),
        # [NOTE] Keep TensorBoard even in ban_all_log mode - it doesn't affect terminal output
        report_to=(tr_cfg.get('report_to', ["tensorboard"]) or ["tensorboard"]),
        remove_unused_columns=False,
        gradient_checkpointing=cfg.get('use_gradient_checkpointing', True),
        gradient_checkpointing_kwargs=gc_kwargs,
        ddp_find_unused_parameters=False,
        warmup_ratio=float(tr_cfg.get('warmup_ratio', 0.0)),
        lr_scheduler_type=tr_cfg.get('lr_scheduler_type', None),
        dataloader_num_workers=tr_cfg.get('dataloader_num_workers', 8),
    )
    if 'per_device_eval_batch_size' in ta_fields and 'per_device_eval_batch_size' in tr_cfg:
        ta_kwargs['per_device_eval_batch_size'] = int(tr_cfg['per_device_eval_batch_size'])
    if 'prediction_loss_only' in ta_fields and 'prediction_loss_only' in tr_cfg:
        ta_kwargs['prediction_loss_only'] = bool(tr_cfg['prediction_loss_only'])
    if 'eval_accumulation_steps' in ta_fields and 'eval_accumulation_steps' in tr_cfg:
        ta_kwargs['eval_accumulation_steps'] = int(tr_cfg['eval_accumulation_steps'])
    if 'fp16_full_eval' in ta_fields and 'fp16_full_eval' in tr_cfg and not use_bf16:
        ta_kwargs['fp16_full_eval'] = bool(tr_cfg['fp16_full_eval'])
    if 'bf16_full_eval' in ta_fields and 'bf16_full_eval' in tr_cfg and use_bf16:
        ta_kwargs['bf16_full_eval'] = bool(tr_cfg['bf16_full_eval'])
    if 'predict_with_generate' in ta_fields and 'predict_with_generate' in tr_cfg:
        ta_kwargs['predict_with_generate'] = bool(tr_cfg['predict_with_generate'])
    if 'include_inputs_for_metrics' in ta_fields and 'include_inputs_for_metrics' in tr_cfg:
        ta_kwargs['include_inputs_for_metrics'] = bool(tr_cfg['include_inputs_for_metrics'])

    supports_eval = 'evaluation_strategy' in ta_fields
    supports_save = 'save_strategy' in ta_fields
    supports_load_best = 'load_best_model_at_end' in ta_fields
    supports_metric = 'metric_for_best_model' in ta_fields
    supports_gib = 'greater_is_better' in ta_fields
    # Support different field names: evaluation_strategy or eval_strategy
    eval_field = 'evaluation_strategy' if 'evaluation_strategy' in ta_fields else ('eval_strategy' if 'eval_strategy' in ta_fields else None)
    supports_eval = eval_field is not None

    # Read strategies from config and align them when both are supported
    eval_strategy_cfg = str(tr_cfg.get('evaluation_strategy', 'steps'))
    save_strategy_cfg = str(tr_cfg.get('save_strategy', eval_strategy_cfg))

    if supports_eval:
        ta_kwargs[eval_field] = eval_strategy_cfg
    if supports_save:
        ta_kwargs['save_strategy'] = eval_strategy_cfg if eval_strategy_cfg != 'no' else 'no'
    else:
        # If save strategy is not supported, ensure best-model logic is not triggered
        if supports_load_best:
            ta_kwargs['load_best_model_at_end'] = False

    # If evaluation field is unsupported, avoid conflicts with save strategy and warn
    if not supports_eval and supports_save:
        ta_kwargs['save_strategy'] = 'no'
        if _is_main():
            print('[EVAL-SETUP] WARNING: This transformers build does not expose evaluation_strategy/eval_strategy; disabling save_strategy to avoid conflicts.')

    # Best-model related
    if supports_load_best:
        load_best_cfg = bool(tr_cfg.get('load_best_model_at_end', True))
        # Only allow when eval/save are not 'no' and eval field is supported
        save_str_current = ta_kwargs.get('save_strategy', 'no') if supports_save else 'no'
        if supports_eval and eval_strategy_cfg != 'no' and save_str_current != 'no':
            ta_kwargs['load_best_model_at_end'] = load_best_cfg
        else:
            ta_kwargs['load_best_model_at_end'] = False

    if supports_metric:
        ta_kwargs['metric_for_best_model'] = str(tr_cfg.get('metric_for_best_model', 'eval_loss'))
    if supports_gib:
        ta_kwargs['greater_is_better'] = bool(tr_cfg.get('greater_is_better', False))

    training_args = TrainingArguments(**ta_kwargs)

    # Configure evaluation to return loss only (save memory)
    try:
        setattr(model if not hasattr(model, 'get_base_model') else model.get_base_model(), 'return_loss_only_eval', bool(getattr(training_args, 'prediction_loss_only', False)))
    except Exception:
        pass

    # Evaluation debug information
    if _is_main():
        # Read strategies (with fallback)
        eval_strategy = getattr(training_args, 'evaluation_strategy', None)
        if eval_strategy is None:
            eval_strategy = getattr(training_args, 'eval_strategy', None)
        save_strategy = getattr(training_args, 'save_strategy', None)
        load_best = bool(getattr(training_args, 'load_best_model_at_end', False))
        per_eval_bs = getattr(training_args, 'per_device_eval_batch_size', None)
        metric_name = getattr(training_args, 'metric_for_best_model', None)
        gib = getattr(training_args, 'greater_is_better', None)
        pred_loss_only = getattr(training_args, 'prediction_loss_only', None)
        eval_accum_steps = getattr(training_args, 'eval_accumulation_steps', None)
        fp16_full_eval = getattr(training_args, 'fp16_full_eval', None)
        pwd = getattr(training_args, 'predict_with_generate', None)
        include_inputs_for_metrics = getattr(training_args, 'include_inputs_for_metrics', None)
        model_loss_only_eval = bool(getattr(model if not hasattr(model, 'get_base_model') else model.get_base_model(), 'return_loss_only_eval', False))
        # Infer whether evaluation will run
        if eval_strategy is None:
            will_eval = (eval_dataset is not None) and (eval_steps > 0)
            eval_strategy_str = 'n/a'
        else:
            eval_strategy_str = str(eval_strategy)
            try:
                eval_strategy_str = eval_strategy.value if hasattr(eval_strategy, 'value') else str(eval_strategy)
            except Exception:
                pass
            will_eval = (eval_dataset is not None) and (eval_strategy_str != 'no')
        save_strategy_str = str(save_strategy) if save_strategy is not None else 'n/a'
        try:
            save_strategy_str = save_strategy.value if hasattr(save_strategy, 'value') else save_strategy_str
        except Exception:
            pass
        print(f"[EVAL-SETUP] enabled={will_eval} eval_strategy={eval_strategy_str} eval_steps={eval_steps} save_strategy={save_strategy_str} load_best={load_best} per_device_eval_batch_size={per_eval_bs} eval_samples={len(eval_dataset)} metric_for_best_model={metric_name} greater_is_better={gib}")
        print(f"[EVAL-SETUP] eval_args: prediction_loss_only={pred_loss_only} eval_accumulation_steps={eval_accum_steps} fp16_full_eval={fp16_full_eval} predict_with_generate={pwd} include_inputs_for_metrics={include_inputs_for_metrics} model_loss_only_eval={model_loss_only_eval}")
        es_cfg_dbg = (cfg.get('early_stopping', {}) or {})
        print(f"[EVAL-SETUP] early_stopping_enable={bool(es_cfg_dbg.get('enable', True))} patience={int(es_cfg_dbg.get('patience', 5))} threshold={float(es_cfg_dbg.get('threshold', 0.0))}")

    # Define whether modulation is disabled (for collate)
    disable_mod = args.disable_modulation

    def collate(features: List[Dict]):
        batch = {}
        if not features:
            return batch
        keys = features[0].keys()
        for k in keys:
            batch[k] = torch.stack([f[k] for f in features])
        
        # Pass YAML inject_* parameters as forward overrides to the model
        try:
            top_inject = {k: v for k, v in cfg.items() if isinstance(k, str) and k.startswith('inject_')}
            nested_inject = {k: v for k, v in (cfg.get('inject', {}) or {}).items() if isinstance(k, str) and k.startswith('inject_')}
            merged = {**top_inject, **nested_inject}
            for k, v in merged.items():
                batch[k] = v
        except Exception:
            pass
        return batch

    class ModulationDebugCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if int(os.environ.get('RANK', '0')) != 0:
                return
            mdl = kwargs.get('model', None)
            if mdl is None:
                return
            if hasattr(mdl, 'get_base_model'):
                mdl = mdl.get_base_model()
            buf = getattr(mdl, '_modulation_buffer', None)
            if not buf:
                s = getattr(mdl, '_last_modulation_stats', None)
                if s is None: return
                self._print_stats(state.global_step, [s])
            else:
                self._print_stats(state.global_step, buf)
                try:
                    mdl._modulation_buffer.clear()
                except Exception:
                    pass

        def _print_stats(self, global_step, stats_list: List[Dict]):
            # Aggregate
            mask_sum = sum(s.get('mask_sum', 0) for s in stats_list)
            valid_tokens = sum(s.get('valid_tokens', 0) for s in stats_list)
            mask_frac_all = sum(s.get('mask_frac', 0.0) for s in stats_list) / max(1, len(stats_list))
            mask_frac_nonpad = (float(mask_sum) / float(valid_tokens)) if valid_tokens > 0 else 0.0
            mean_abs_delta = sum(s.get('mean_abs_delta', 0.0) for s in stats_list) / max(1, len(stats_list))
            max_abs_delta = max(s.get('max_abs_delta', 0.0) for s in stats_list)
            masked_mean_abs_delta = sum(s.get('masked_mean_abs_delta', 0.0) for s in stats_list) / max(1, len(stats_list))
            # Merge several examples (deduplicate)
            samples = []
            seen = set()
            for s in stats_list:
                for it in s.get('samples', [])[:4]:
                    key = (it.get('mb', 0), it['b'], it['pos'], it['id'])
                    if key in seen: continue
                    seen.add(key)
                    samples.append(it)
                    if len(samples) >= 8:
                        break
                if len(samples) >= 8:
                    break
            base = f"[MOD] step={int(global_step)} mask_frac_nonpad={mask_frac_nonpad:.4f} mask_frac={mask_frac_all:.4f} mask_sum={mask_sum} mean|Δemb|={mean_abs_delta:.6f} masked_mean|Δemb|={masked_mean_abs_delta:.6f} max|Δemb|={max_abs_delta:.6f}"
            if samples:
                head = "; ".join([f"b{it['b']}@{it['pos']} id={it['id']} tok={it['tok']}" for it in samples])
                print(base + " | examples: " + head)
            else:
                print(base)

    class EvalPPLCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if not _is_main() or not metrics:
                return
            loss = metrics.get('eval_loss', None)
            if loss is None:
                return
            try:
                ppl = math.exp(float(loss))
                print(f"[EVAL] step={int(state.global_step)} eval_loss={float(loss):.4f} ppl={ppl:.2f}")
            except Exception:
                pass

    class CustomTensorBoardCallback(TrainerCallback):
        def __init__(self):
            self.writer = None
        def _is_world_zero(self, args) -> bool:
            try:
                return getattr(args, 'process_index', 0) == 0
            except Exception:
                return _is_main()
        def _get_base(self, model):
            if model is None:
                return None
            return model.get_base_model() if hasattr(model, 'get_base_model') else model
        def on_train_begin(self, args, state, control, **kwargs):
            # Keep placeholder but do not write to avoid conflicts with the HF default writer
            return
        def on_log(self, args, state, control, logs=None, model=None, **kwargs):
            return
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            return
        def on_train_end(self, args, state, control, **kwargs):
            return

    class GradientDiagnosticCallback(TrainerCallback):
        """Diagnose LoRA parameter gradient distributions to identify causes of near-zero grad_norm"""
        def __init__(self, max_steps: int = 30):
            self.max_steps = max_steps
            self._counter = 0
        
        def on_step_end(self, args, state, control, model=None, **kwargs):
            if not _is_main() or self._counter >= self.max_steps:
                return
            
            if model is None:
                return
            
            step = state.global_step
            
            # Collect statistics for all parameters that have gradients
            lora_grads = []
            non_lora_grads = []
            zero_grad_params = []
            none_grad_params = []
            
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                if param.grad is None:
                    none_grad_params.append(name)
                    continue
                
                g = param.grad
                g_norm = g.norm().item()
                g_mean = g.abs().mean().item()
                g_max = g.abs().max().item()
                g_zero_frac = (g == 0).float().mean().item()
                
                info = {
                    'name': name,
                    'norm': g_norm,
                    'mean': g_mean,
                    'max': g_max,
                    'zero_frac': g_zero_frac,
                    'shape': list(g.shape),
                }
                
                if 'lora' in name.lower():
                    lora_grads.append(info)
                else:
                    non_lora_grads.append(info)
                
                if g_norm < 1e-8:
                    zero_grad_params.append(name)
            
            # Print summary
            print(f"\n[GRAD-SUMMARY] step={step}")
            
            if none_grad_params:
                print(f"  [WARN] {len(none_grad_params)} params have grad=None: {none_grad_params[:5]}...")
            
            if zero_grad_params:
                print(f"  [WARN] {len(zero_grad_params)} params have near-zero grad: {zero_grad_params[:5]}...")
            
            # LoRA gradient statistics
            if lora_grads:
                total_lora_norm = sum(g['norm']**2 for g in lora_grads) ** 0.5
                avg_lora_mean = sum(g['mean'] for g in lora_grads) / len(lora_grads)
                max_lora_max = max(g['max'] for g in lora_grads)
                avg_zero_frac = sum(g['zero_frac'] for g in lora_grads) / len(lora_grads)
                
                print(f"  [LORA] n_params={len(lora_grads)} total_norm={total_lora_norm:.6f} avg_mean={avg_lora_mean:.6e} max_max={max_lora_max:.6e} avg_zero_frac={avg_zero_frac:.4f}")
                
                # Print top 3 LoRA parameters by gradient norm
                top3 = sorted(lora_grads, key=lambda x: x['norm'], reverse=True)[:3]
                for g in top3:
                    print(f"    TOP: {g['name'][-50:]} norm={g['norm']:.6f} mean={g['mean']:.6e}")
                
                # Print bottom 3 LoRA parameters by gradient norm
                bot3 = sorted(lora_grads, key=lambda x: x['norm'])[:3]
                for g in bot3:
                    print(f"    BOT: {g['name'][-50:]} norm={g['norm']:.6e} mean={g['mean']:.6e}")
            
            # Non-LoRA gradients (e.g., embedding modulation)
            if non_lora_grads:
                total_other_norm = sum(g['norm']**2 for g in non_lora_grads) ** 0.5
                print(f"  [OTHER] n_params={len(non_lora_grads)} total_norm={total_other_norm:.6f}")
                for g in non_lora_grads[:3]:
                    print(f"    {g['name'][-50:]} norm={g['norm']:.6f} mean={g['mean']:.6e}")
            
            self._counter += 1

    # =========================================================================
    # DP-SGD Privacy Accounting Callback
    # =========================================================================
    class DPPrivacyCallback(TrainerCallback):
        """Callback: periodically record DP-SGD privacy spend (ε, δ)"""
        def __init__(self, privacy_engine, dp_config, dataset_size):
            self.privacy_engine = privacy_engine
            self.dp_config = dp_config
            self.dataset_size = dataset_size
            self.log_every = int(dp_config.get('log_epsilon_every_n_steps', 50))
            self.delta = float(dp_config.get('delta', 1e-5))
            self.target_epsilon = dp_config.get('target_epsilon', None)
            self._step_counter = 0
        
        def on_step_end(self, args, state, control, **kwargs):
            if self.privacy_engine is None:
                return
            
            self._step_counter += 1
            if self._step_counter % self.log_every != 0:
                return
            
            try:
                epsilon = self.privacy_engine.get_epsilon(self.delta)
                
                if _is_main():
                    print(f"[DP-SGD] Step {state.global_step}: (ε, δ) = ({epsilon:.4f}, {self.delta})")
                
                # Check whether the target has been exceeded
                if self.target_epsilon is not None and epsilon > float(self.target_epsilon):
                    if _is_main():
                        print(f"[DP-SGD] ⚠️ Exceeded target epsilon! Current={epsilon:.4f} > Target={self.target_epsilon}")
                    # Optional: set control.should_training_stop = True to stop training
            except Exception as e:
                if _is_main():
                    print(f"[DP-SGD] Privacy accounting error: {e}")
        
        def on_train_end(self, args, state, control, **kwargs):
            if self.privacy_engine is None:
                return
            
            try:
                epsilon = self.privacy_engine.get_epsilon(self.delta)
                if _is_main():
                    print("\n" + "="*70)
                    print("[DP-SGD] 🔒 FINAL PRIVACY REPORT")
                    print("="*70)
                    print(f"  Total training steps: {state.global_step}")
                    print(f"  Dataset size: {self.dataset_size}")
                    print(f"  Final (ε, δ): ({epsilon:.4f}, {self.delta})")
                    print(f"  Noise multiplier: {self.dp_config.get('noise_multiplier', 'N/A')}")
                    print(f"  Max grad norm: {self.dp_config.get('max_grad_norm', 'N/A')}")
                    print("="*70)
                    print(f"  ✅ This model satisfies ({epsilon:.2f}, {self.delta})-differential privacy")
                    print("="*70 + "\n")
            except Exception as e:
                if _is_main():
                    print(f"[DP-SGD] Error in final privacy report: {e}")

    # Register callbacks
    callbacks = []
    if getattr(args, 'debug_modulation', False):
        callbacks.append(ModulationDebugCallback())
    
    # [STEALTH] Skip debug callbacks if ban_all_log is enabled
    if not ban_all_log:
        # [FIX] Disable GradientDiagnosticCallback when running DP-SGD
        # Because Opacus clears gradients after optimizer.step(), which can trigger false "grad=None" reports
        if not dp_enabled:
            callbacks.append(GradientDiagnosticCallback(max_steps=30))
    
    es_cfg = cfg.get('early_stopping', {}) or {}
    if es_cfg.get('enable', True):
        es_patience = int(es_cfg.get('patience', 5))
        es_threshold = float(es_cfg.get('threshold', 0.0))
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=es_patience, early_stopping_threshold=es_threshold))
    
    # [STEALTH] Skip TensorBoard and diagnostic callbacks if ban_all_log is enabled
    if not ban_all_log:
        callbacks.append(EvalPPLCallback())
        exp_tb_cb = CustomExpTensorBoardCallback()
        callbacks.append(exp_tb_cb)
        loss_bd_cb = ExactLossBreakdownCallback()
        callbacks.append(loss_bd_cb)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        data_collator=collate,
        callbacks=callbacks,
        compute_metrics=None,
    )


    # [NEW] Get eval_num_samples from config
    eval_num_samples = int(tr_cfg.get('eval_num_samples', 50))

    if cfg.get('data', {}).get('type') == 'aeslc':
        from src.utils.evaluation import AESLCGenerationCallback
        gen_cb = AESLCGenerationCallback(
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            trainer=trainer,
            num_samples=eval_num_samples,
            max_prompt_length=max_length  # max_length from config
        )
        trainer.add_callback(gen_cb)
        # pass

    if cfg.get('data', {}).get('type') == 'xsum':
        from src.utils.evaluation import XSumGenerationCallback
        gen_cb = XSumGenerationCallback(
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            trainer=trainer,
            num_samples=eval_num_samples,
            max_prompt_length=max_length  # max_length from config
        )
        trainer.add_callback(gen_cb)

    if cfg.get('data', {}).get('type') == 'healthcaremagic':
        from src.utils.evaluation import HealthCareMagicGenerationCallback
        gen_cb = HealthCareMagicGenerationCallback(
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            trainer=trainer,
            num_samples=eval_num_samples,
            max_prompt_length=max_length  # max_length from config
        )
        trainer.add_callback(gen_cb)

    if cfg.get('data', {}).get('type') == 'cuadqa':
        from src.utils.evaluation import CUADQAGenerationCallback
        gen_cb = CUADQAGenerationCallback(
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            trainer=trainer,
            num_samples=eval_num_samples,
            max_prompt_length=max_length  # max_length from config
        )
        trainer.add_callback(gen_cb)

    if cfg.get('data', {}).get('type') == 'magicoder':
        from src.utils.evaluation import MagicoderGenerationCallback
        gen_cb = MagicoderGenerationCallback(
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            trainer=trainer,
            num_samples=eval_num_samples,
            max_prompt_length=max_length  # max_length from config
        )
        trainer.add_callback(gen_cb)

    # [STEALTH] Only attach TensorBoard callbacks if not banned
    if not ban_all_log:
        # Attach Trainer reference to custom callbacks
        exp_tb_cb.attach_trainer(trainer)
        loss_bd_cb.attach_trainer(trainer)
        
        # Ensure breakdown callbacks appear first
        handler = getattr(trainer, 'callback_handler', None)
        if handler is not None:
            for cb in (loss_bd_cb, exp_tb_cb):
                for attr in ('callbacks', 'callback_list'):
                    lst = getattr(handler, attr, None)
                    if isinstance(lst, list) and cb in lst:
                        try:
                            lst.remove(cb)
                            lst.insert(0, cb)
                        except Exception:
                            pass
        
    # If gradient checkpointing is enabled, disable use_cache
    if training_args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()

    # [CRITICAL FIX] Create a callback to set static graph on DDP before first training step
    # This is needed because the surrogate loss injection creates multiple gradient paths to LoRA parameters.
    class SetStaticGraphCallback(TrainerCallback):
        def __init__(self, trainer_ref):
            self._done = False
            self._trainer = trainer_ref
        
        def on_step_begin(self, args, state, control, model=None, **kwargs):
            if self._done:
                return
            self._done = True
            
            try:
                from torch.nn.parallel import DistributedDataParallel as DDP
                
                # Access the model through accelerator which has the actual DDP wrapper
                if hasattr(self._trainer, 'accelerator') and self._trainer.accelerator is not None:
                    # The accelerator's _models list contains the wrapped models
                    for m in getattr(self._trainer.accelerator, '_models', []):
                        if isinstance(m, DDP):
                            m._set_static_graph()
                            if _is_main():
                                print("[DDP-FIX] Successfully set static graph on accelerator DDP model")
                            return
                    
                    # Also check the trainer.model directly
                    trainer_model = self._trainer.model
                    if isinstance(trainer_model, DDP):
                        trainer_model._set_static_graph()
                        if _is_main():
                            print("[DDP-FIX] Successfully set static graph on trainer.model (DDP)")
                        return
                
                # Fallback: traverse model hierarchy
                def find_ddp(m, depth=0):
                    if depth > 10:
                        return None
                    if isinstance(m, DDP):
                        return m
                    if hasattr(m, 'module'):
                        result = find_ddp(m.module, depth + 1)
                        if result:
                            return result
                    return None
                
                target = find_ddp(self._trainer.model)
                if target is not None:
                    target._set_static_graph()
                    if _is_main():
                        print("[DDP-FIX] Successfully set static graph via model hierarchy")
                else:
                    if _is_main():
                        import torch.distributed as dist
                        if not dist.is_initialized() or dist.get_world_size() == 1:
                            print("[DDP-FIX] Single GPU mode - DDP may use different wrapper")
                        print(f"[DDP-FIX] trainer.model type: {type(self._trainer.model)}")
                        if hasattr(self._trainer, 'accelerator'):
                            print(f"[DDP-FIX] accelerator._models: {[type(m).__name__ for m in getattr(self._trainer.accelerator, '_models', [])]}")
                            
            except Exception as e:
                if _is_main():
                    import traceback
                    print(f"[DDP-FIX] Warning: Could not set static graph: {e}")
                    traceback.print_exc()
    
    # Note: We need to add callback after trainer is created but before train() is called
    static_graph_cb = SetStaticGraphCallback(trainer)
    trainer.add_callback(static_graph_cb)
    
    # =========================================================================
    # DP-SGD Setup with Opacus
    # =========================================================================
    dp_cfg = cfg.get('dp_sgd', {}) or {}
    dp_enabled = bool(dp_cfg.get('enable', False)) and not getattr(args, 'disable_dp', False)
    privacy_engine = None
    
    if dp_enabled:
        if not OPACUS_AVAILABLE:
            raise ImportError(
                "[DP-SGD] Opacus is not installed! Please run: pip install opacus>=1.4.0"
            )
        
        if _is_main():
            print("\n" + "="*70)
            print("[DP-SGD] Differential Privacy Training Enabled")
            print("="*70)
        
        # Read DP-SGD configuration
        noise_multiplier = float(dp_cfg.get('noise_multiplier', 1.0))
        dp_max_grad_norm = float(dp_cfg.get('max_grad_norm', 1.0))
        delta = float(dp_cfg.get('delta', 1e-5))
        secure_mode = bool(dp_cfg.get('secure_mode', False))
        target_epsilon = dp_cfg.get('target_epsilon', None)
        log_epsilon_every_n_steps = int(dp_cfg.get('log_epsilon_every_n_steps', 50))
        
        if _is_main():
            print(f"[DP-SGD] noise_multiplier (σ) = {noise_multiplier}")
            print(f"[DP-SGD] max_grad_norm (C) = {dp_max_grad_norm}")
            print(f"[DP-SGD] delta (δ) = {delta}")
            print(f"[DP-SGD] secure_mode = {secure_mode}")
            print(f"[DP-SGD] target_epsilon = {target_epsilon}")
            print(f"[DP-SGD] Dataset size = {len(dataset)}")
        
        # Validate configuration constraints
        if cfg.get('use_gradient_checkpointing', False):
            raise ValueError(
                "[DP-SGD] Gradient checkpointing is NOT compatible with Opacus! "
                "Please set use_gradient_checkpointing: false in your config."
            )
        
        if int(tr_cfg.get('gradient_accumulation_steps', 1)) > 1:
            if _is_main():
                print("[DP-SGD] WARNING: gradient_accumulation_steps > 1 is not recommended with DP-SGD!")
                print("[DP-SGD] Setting gradient_accumulation_steps = 1 for DP training.")
        
        # Get the model (must be the base model, not the PEFT wrapper)
        # Opacus needs direct access to the model's forward hooks
        dp_model = trainer.model
        
        # Validate model compatibility using ModuleValidator
        try:
            from opacus.validators import ModuleValidator
            errors = ModuleValidator.validate(dp_model, strict=False)
            if errors:
                if _is_main():
                    print(f"[DP-SGD] ModuleValidator found {len(errors)} issues:")
                    for e in errors[:5]:
                        print(f"  - {e}")
                # Try to fix model issues
                dp_model = ModuleValidator.fix(dp_model)
                if _is_main():
                    print("[DP-SGD] Applied ModuleValidator.fix() to model")
        except Exception as e:
            if _is_main():
                print(f"[DP-SGD] ModuleValidator check failed (non-fatal): {e}")
        
        # Create PrivacyEngine
        privacy_engine = PrivacyEngine(
            accountant='rdp',
            secure_mode=secure_mode,
        )
        
        # Get Trainer's dataloader and optimizer
        # Note: must be obtained when called inside Trainer; we create a custom training loop here
        
        if _is_main():
            print("[DP-SGD] Privacy Engine created successfully")
            print("="*70 + "\n")
        
        # =====================================================================
        # Create DP-SGD-aware Trainer subclass
        # =====================================================================
        class DPSGDTrainer(Trainer):
            """
            Custom Trainer integrating Opacus PrivacyEngine to perform DP-SGD training.
            """
            def __init__(self, privacy_engine, dp_config, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._privacy_engine = privacy_engine
                self._dp_config = dp_config
                self._dp_attached = False
                self._epsilon_log_counter = 0
                
            def _maybe_attach_privacy_engine(self):
                """Attach PrivacyEngine before the first training step"""
                if self._dp_attached:
                    return
                
                noise_mult = float(self._dp_config.get('noise_multiplier', 1.0))
                max_gn = float(self._dp_config.get('max_grad_norm', 1.0))
                
                # Get train dataloader
                train_dl = self.get_train_dataloader()
                
                # Attach PrivacyEngine
                # This will wrap model, optimizer, and dataloader
                # Use poisson_sampling=False to avoid conflicts with Trainer internals
                self.model, self.optimizer, self._dp_train_dataloader = \
                    self._privacy_engine.make_private(
                        module=self.model,
                        optimizer=self.optimizer,
                        data_loader=train_dl,
                        noise_multiplier=noise_mult,
                        max_grad_norm=max_gn,
                        poisson_sampling=False,  # [FIX] Disable Poisson sampling to use fixed batches
                    )
                
                self._dp_attached = True
                
                if _is_main():
                    print(f"[DP-SGD] PrivacyEngine attached to model and optimizer")
                    print(f"[DP-SGD] Model wrapped with GradSampleModule")
                
                # [FIX] Re-sync configuration to the base model (after Opacus wrapping)
                try:
                    unwrapped = _unwrap_model(self.model)
                    if unwrapped is not None and hasattr(unwrapped, 'config'):
                        # Cannot obtain original yaml cfg from self.args (TrainingArguments)
                        # But we can read saved values from the model's config and verify their presence
                        if _is_main():
                            debug_enabled = getattr(unwrapped.config, 'super_aux_example_debug', None)
                            debug_file = getattr(unwrapped.config, 'super_aux_example_debug_file', None)
                            print(f"[DP-SGD-CFG-CHECK] super_aux_example_debug={debug_enabled}")
                            if debug_file:
                                print(f"[DP-SGD-CFG-CHECK] super_aux_example_debug_file={debug_file}")
                except Exception as e:
                    if _is_main():
                        print(f"[DP-SGD-CFG-CHECK] Warning: {e}")
            
            def get_train_dataloader(self):
                """Return DP-wrapped dataloader (if attached)"""
                if self._dp_attached and hasattr(self, '_dp_train_dataloader'):
                    return self._dp_train_dataloader
                return super().get_train_dataloader()
            
            def training_step(self, model, inputs, num_items_in_batch=None):
                """Override training step to handle DP"""
                # Ensure PrivacyEngine is attached
                if not self._dp_attached:
                    self._maybe_attach_privacy_engine()
                
                # [FIX] Opacus requires clearing grad_sample before each step
                # This must be done before forward/backward, otherwise DPOptimizer.step() will error
                if self._dp_attached and self.optimizer is not None:
                    self.optimizer.zero_grad()
                
                # Call parent class training_step
                loss = super().training_step(model, inputs, num_items_in_batch)
                
                # Periodically log epsilon
                log_every = int(self._dp_config.get('log_epsilon_every_n_steps', 50))
                self._epsilon_log_counter += 1
                
                if self._epsilon_log_counter % log_every == 0:
                    self._log_privacy_spent()
                
                return loss
            
            def _log_privacy_spent(self):
                """Log current privacy spent"""
                if not self._dp_attached or self._privacy_engine is None:
                    return
                
                try:
                    delta = float(self._dp_config.get('delta', 1e-5))
                    epsilon = self._privacy_engine.get_epsilon(delta)
                    
                    if _is_main():
                        step = self.state.global_step if hasattr(self, 'state') else '?'
                        print(f"[DP-SGD] Step {step}: Current (ε, δ) = ({epsilon:.4f}, {delta})")
                    
                    # Log to TensorBoard
                    if hasattr(self, 'state') and self.state is not None:
                        self.log({'dp_epsilon': epsilon, 'dp_delta': delta})
                    
                    # Check if target epsilon exceeded
                    target_eps = self._dp_config.get('target_epsilon', None)
                    if target_eps is not None and epsilon > float(target_eps):
                        if _is_main():
                            print(f"[DP-SGD] WARNING: Exceeded target epsilon! "
                                  f"Current ε={epsilon:.4f} > target={target_eps}")
                            print("[DP-SGD] Consider stopping training to preserve privacy budget.")
                except Exception as e:
                    if _is_main():
                        print(f"[DP-SGD] Error computing epsilon: {e}")
            
            def train(self, *args, **kwargs):
                """Override train to attach PrivacyEngine before training starts"""
                # Create optimizer (if not yet created)
                if self.optimizer is None:
                    self.create_optimizer()
                
                # Attach PrivacyEngine
                self._maybe_attach_privacy_engine()
                
                if _is_main():
                    print("\n[DP-SGD] Starting differential privacy training...")
                    print(f"[DP-SGD] Tracking privacy budget with δ = {self._dp_config.get('delta', 1e-5)}")
                    # Show bypass mode patch status
                    try:
                        from src.models.modulated_llama_dpsgd import _DPOPTIMIZER_PATCHED, _AUX_GRAD_MODEL_REGISTRY
                        print(f"[DP-SGD] AUX-BYPASS pre_step patch: {'ACTIVE ✓' if _DPOPTIMIZER_PATCHED else 'NOT ACTIVE ✗'}")
                    except Exception:
                        pass
                
                # Call parent class train
                result = super().train(*args, **kwargs)
                
                # Print final privacy budget after training completes
                self._log_final_privacy()
                
                return result
            
            def _log_final_privacy(self):
                """Log final privacy consumption after training"""
                if not self._dp_attached or self._privacy_engine is None:
                    return
                
                try:
                    delta = float(self._dp_config.get('delta', 1e-5))
                    epsilon = self._privacy_engine.get_epsilon(delta)
                    
                    if _is_main():
                        print("\n" + "="*70)
                        print("[DP-SGD] Training Complete - Final Privacy Report")
                        print("="*70)
                        print(f"[DP-SGD] Final (ε, δ) = ({epsilon:.4f}, {delta})")
                        print(f"[DP-SGD] Interpretation: This model provides ({epsilon:.2f}, {delta})-differential privacy")
                        print("="*70 + "\n")
                except Exception as e:
                    if _is_main():
                        print(f"[DP-SGD] Error computing final epsilon: {e}")
        
        # Replace original trainer with DPSGDTrainer
        dp_trainer = DPSGDTrainer(
            privacy_engine=privacy_engine,
            dp_config=dp_cfg,
            model=model,
            args=training_args,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
            data_collator=collate,
            callbacks=callbacks,
            compute_metrics=None,
        )
        
        # Re-add evaluation callbacks
        if cfg.get('data', {}).get('type') == 'aeslc':
            from src.utils.evaluation import AESLCGenerationCallback
            gen_cb = AESLCGenerationCallback(
                eval_dataset=eval_dataset, tokenizer=tokenizer,
                trainer=dp_trainer, num_samples=eval_num_samples,
                max_prompt_length=max_length
            )
            dp_trainer.add_callback(gen_cb)
        
        # Use DP trainer
        trainer = dp_trainer
        
        if _is_main():
            print("[DP-SGD] DPSGDTrainer initialized and ready")
    
    # =========================================================================
    # Start Training
    # =========================================================================
    trainer.train()
    
    # [MODIFIED] Save to a data_type subdirectory (use out_dir/final_model)
    final_save_path = os.path.join(out_dir, "final_model")
    if _is_main():
        print(f"[SAVE] Saving final model to {final_save_path}")
    
    # [FIX] Opacus GradSampleModule wrapping can cause tensor sharing issues
    # Need to unwrap the model before saving, or use PEFT's save methods
    try:
        # Try to obtain the base model (unwrap Opacus and PEFT wrappers)
        model_to_save = trainer.model
        
        # Unwrap Opacus GradSampleModule
        if hasattr(model_to_save, '_module'):
            model_to_save = model_to_save._module
            if _is_main():
                print("[SAVE] Unwrapped Opacus GradSampleModule")
        
        # If it's a PEFT model, use PEFT's save_pretrained method
        if hasattr(model_to_save, 'save_pretrained'):
            model_to_save.save_pretrained(final_save_path)
            if _is_main():
                print(f"[SAVE] Model saved using save_pretrained to {final_save_path}")
        else:
            # Fallback: save state_dict
            os.makedirs(final_save_path, exist_ok=True)
            torch.save(model_to_save.state_dict(), os.path.join(final_save_path, "pytorch_model.bin"))
            if _is_main():
                print(f"[SAVE] Model state_dict saved to {final_save_path}/pytorch_model.bin")
    except Exception as e:
        if _is_main():
            print(f"[SAVE-WARN] Failed to save model: {e}")
            print("[SAVE-WARN] Trying fallback save method...")
        try:
            # Fallback: directly save state_dict, skip safetensors
            os.makedirs(final_save_path, exist_ok=True)
            state_dict = {k: v for k, v in trainer.model.state_dict().items()}
            torch.save(state_dict, os.path.join(final_save_path, "pytorch_model.bin"))
            if _is_main():
                print(f"[SAVE] Fallback: state_dict saved to {final_save_path}/pytorch_model.bin")
        except Exception as e2:
            if _is_main():
                print(f"[SAVE-ERROR] All save methods failed: {e2}")
    
    tokenizer.save_pretrained(final_save_path)

if __name__ == '__main__':
    main()
