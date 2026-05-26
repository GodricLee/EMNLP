#!/usr/bin/env python3
import os, sys, argparse, yaml, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# [NEW] Check Flash Attention 2 availability
try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
    FLASH_ATTN_VERSION = getattr(flash_attn, '__version__', 'unknown')
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    FLASH_ATTN_VERSION = None
from typing import List, Dict
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer, 
    set_seed, 
    TrainerCallback, 
    EarlyStoppingCallback
)
import math
from peft import get_peft_model, LoraConfig, TaskType

# Reuse existing data modules
from src.data.adapters import get_adapter
from src.data.dataset import UnifiedChatDataset

# Print only on the main process
def _is_main() -> bool:
    return str(os.environ.get('RANK', '0')) == '0'

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

# Configure Transformers logging level: info on main, error on workers
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
    # remove extension
    name_no_ext = os.path.splitext(filename)[0]
    
    if '_' in name_no_ext:
        return name_no_ext.split('_')[-1]
    else:
        return name_no_ext

def parse_args():
    ap = argparse.ArgumentParser(description='Train Baseline Llama (No Modulation)')
    ap.add_argument('--config', default='configs/default.yaml')
    return ap.parse_args()

def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get('seed', 42))
    model_name = cfg['model_name']
    max_length = cfg.get('max_length', 512)
    
    # Get data type
    data_type = cfg.get('data', {}).get('type', 'legacy')

    # [NEW] Flash Attention 2 support - same as train_modulated.py
    use_flash_attn = cfg.get('use_flash_attention_2', False)
    attn_impl = None
    if use_flash_attn:
        if FLASH_ATTN_AVAILABLE:
            attn_impl = "flash_attention_2"
            if _is_main():
                print(f"[FLASH-ATTN] Enabling Flash Attention 2 (version {FLASH_ATTN_VERSION})")
        else:
            if _is_main():
                print("[FLASH-ATTN] WARNING: Flash Attention 2 requested but flash-attn not installed. Using default attention.")

    # 1. Load Standard Model (No Modulation)
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else None,
    }
    if attn_impl:
        model_kwargs["attn_implementation"] = attn_impl
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token else '<pad>'
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if model.get_input_embeddings().weight.size(0) < len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # 3. LoRA
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
        # PEFT initializes LoRA weights as float32 by default, causing dtype mismatch
        # and slower mixed-precision training. Converting to bfloat16 improves performance.
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

    # 4. Data Loading
    adapter = get_adapter(cfg)
    train_samples = adapter.load_data('train')
    val_samples = adapter.load_data('validation')

    dataset = UnifiedChatDataset(train_samples, tokenizer, max_length)
    eval_dataset = UnifiedChatDataset(val_samples, tokenizer, max_length)
    
    if _is_main():
        print(f"[DATA] train_samples={len(dataset)} val_samples={len(eval_dataset)}")

    if len(dataset) == 0:
        raise ValueError("Train dataset is empty! Please check your data path and config.")

    # 5. Training Arguments
    tr_cfg = cfg['training']
    per_dev_bs = int(tr_cfg['per_device_train_batch_size'])
    grad_accum = int(tr_cfg['gradient_accumulation_steps'])
    
    # [MODIFIED] Dynamic output path adjustments
    # Prepare variables
    train_suffix = _get_train_file_suffix(cfg, data_type)
    inject_cfg = cfg.get('inject', {}) or {}
    expand_tokens = inject_cfg.get('inject_per_sample_total_expand_tokens', 0)
    
    raw_output_dir = tr_cfg.get('output_dir', 'baseline_output')
    
    # Variable substitutions
    # [FIX] Append _baseline to data_type to avoid collision with train_modulated output
    processed_output_dir = raw_output_dir.replace('$DATA_TYPE', f"{data_type}_baseline")
    processed_output_dir = processed_output_dir.replace('$train_file_suffix', str(train_suffix))
    processed_output_dir = processed_output_dir.replace('$inject_per_sample_total_expand_tokens', str(expand_tokens))
    
    base_out_dir = _expand_dynamic_path(processed_output_dir)
    
    # Backwards compatibility: if template did not include $DATA_TYPE, append data_type subdir
    if '$DATA_TYPE' not in raw_output_dir:
        out_dir = os.path.join(base_out_dir, f"{data_type}_baseline")
    else:
        out_dir = base_out_dir

    # Print effective batch size
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    eff_bs = per_dev_bs * max(1, world_size) * grad_accum
    if _is_main():
        print(f"[ARGS] per_device_train_batch_size={per_dev_bs} grad_accum={grad_accum} world_size={world_size} -> effective_batch_size={eff_bs} output_dir={out_dir}")

    eval_steps = int(tr_cfg.get('eval_steps', 200))
    save_steps = int(tr_cfg.get('save_steps', eval_steps))

    # [NEW] TF32 support - same as train_modulated.py
    use_tf32 = bool(tr_cfg.get('tf32', False)) if torch.cuda.is_available() else False
    if use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if _is_main():
            print("[TF32] Enabled TF32 for matmul and cudnn")

    # [MODIFIED] bf16 instead of fp16 - same as train_modulated.py
    use_bf16 = bool(tr_cfg.get('bf16', False)) if torch.cuda.is_available() else False
    use_fp16 = bool(tr_cfg.get('fp16', False)) if torch.cuda.is_available() else False
    # bf16 takes priority over fp16
    if use_bf16:
        use_fp16 = False

    # Construct TrainingArguments (copied logic for consistency)
    ta_fields = set(getattr(TrainingArguments, '__dataclass_fields__', {}).keys())
    
    # [FIX] Add use_reentrant=False to match train_modulated.py for consistent gradient behavior
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
        report_to=(tr_cfg.get('report_to', ["tensorboard"]) or ["tensorboard"]),
        remove_unused_columns=False,
        gradient_checkpointing=cfg.get('use_gradient_checkpointing', True),
        gradient_checkpointing_kwargs=gc_kwargs,  # [NEW] Match modulated training
        ddp_find_unused_parameters=False,
        warmup_ratio=float(tr_cfg.get('warmup_ratio', 0.0)),
        lr_scheduler_type=tr_cfg.get('lr_scheduler_type', None),
        dataloader_num_workers=tr_cfg.get('dataloader_num_workers', 8),
    )

    # [NEW] tf32 in TrainingArguments if supported
    if 'tf32' in ta_fields:
        ta_kwargs['tf32'] = use_tf32
    
    if 'per_device_eval_batch_size' in ta_fields and 'per_device_eval_batch_size' in tr_cfg:
        ta_kwargs['per_device_eval_batch_size'] = int(tr_cfg['per_device_eval_batch_size'])
    if 'prediction_loss_only' in ta_fields and 'prediction_loss_only' in tr_cfg:
        ta_kwargs['prediction_loss_only'] = bool(tr_cfg['prediction_loss_only'])
    if 'eval_accumulation_steps' in ta_fields and 'eval_accumulation_steps' in tr_cfg:
        ta_kwargs['eval_accumulation_steps'] = int(tr_cfg['eval_accumulation_steps'])
    # [MODIFIED] bf16_full_eval instead of fp16_full_eval - same as train_modulated.py
    if 'bf16_full_eval' in ta_fields and 'bf16_full_eval' in tr_cfg:
        ta_kwargs['bf16_full_eval'] = bool(tr_cfg['bf16_full_eval'])
    if 'fp16_full_eval' in ta_fields and 'fp16_full_eval' in tr_cfg:
        ta_kwargs['fp16_full_eval'] = bool(tr_cfg['fp16_full_eval'])
    if 'predict_with_generate' in ta_fields and 'predict_with_generate' in tr_cfg:
        ta_kwargs['predict_with_generate'] = bool(tr_cfg['predict_with_generate'])
    if 'include_inputs_for_metrics' in ta_fields and 'include_inputs_for_metrics' in tr_cfg:
        ta_kwargs['include_inputs_for_metrics'] = bool(tr_cfg['include_inputs_for_metrics'])

    supports_eval = 'evaluation_strategy' in ta_fields
    supports_save = 'save_strategy' in ta_fields
    supports_load_best = 'load_best_model_at_end' in ta_fields
    supports_metric = 'metric_for_best_model' in ta_fields
    supports_gib = 'greater_is_better' in ta_fields
    
    eval_field = 'evaluation_strategy' if 'evaluation_strategy' in ta_fields else ('eval_strategy' if 'eval_strategy' in ta_fields else None)
    supports_eval = eval_field is not None

    eval_strategy_cfg = str(tr_cfg.get('evaluation_strategy', 'steps'))
    
    if supports_eval:
        ta_kwargs[eval_field] = eval_strategy_cfg
    if supports_save:
        ta_kwargs['save_strategy'] = eval_strategy_cfg if eval_strategy_cfg != 'no' else 'no'
    else:
        if supports_load_best:
            ta_kwargs['load_best_model_at_end'] = False

    if not supports_eval and supports_save:
        ta_kwargs['save_strategy'] = 'no'

    if supports_load_best:
        load_best_cfg = bool(tr_cfg.get('load_best_model_at_end', True))
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

    # Inject return_loss_only_eval attribute to match the modulated version
    try:
        setattr(model if not hasattr(model, 'get_base_model') else model.get_base_model(), 'return_loss_only_eval', bool(getattr(training_args, 'prediction_loss_only', False)))
    except Exception:
        pass

    # Evaluation debug info (match modulated version)
    if _is_main():
        eval_strategy = getattr(training_args, 'evaluation_strategy', None) or getattr(training_args, 'eval_strategy', None)
        save_strategy = getattr(training_args, 'save_strategy', None)
        load_best = bool(getattr(training_args, 'load_best_model_at_end', False))
        per_eval_bs = getattr(training_args, 'per_device_eval_batch_size', None)
        metric_name = getattr(training_args, 'metric_for_best_model', None)
        gib = getattr(training_args, 'greater_is_better', None)
        
        eval_strategy_str = str(eval_strategy) if eval_strategy is not None else 'n/a'
        save_strategy_str = str(save_strategy) if save_strategy is not None else 'n/a'
        
        will_eval = (eval_dataset is not None) and (eval_strategy_str != 'no') and (eval_strategy_str != 'n/a')
        
        print(f"[EVAL-SETUP] enabled={will_eval} eval_strategy={eval_strategy_str} eval_steps={eval_steps} save_strategy={save_strategy_str} load_best={load_best} per_device_eval_batch_size={per_eval_bs} eval_samples={len(eval_dataset)} metric_for_best_model={metric_name} greater_is_better={gib}")

    # 6. Callbacks
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

    callbacks = [EvalPPLCallback()]
    
    es_cfg = cfg.get('early_stopping', {}) or {}
    if es_cfg.get('enable', True):
        es_patience = int(es_cfg.get('patience', 5))
        es_threshold = float(es_cfg.get('threshold', 0.0))
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=es_patience, early_stopping_threshold=es_threshold))

    # 7. Collate Function
    def collate(features: List[Dict]):
        batch = {}
        # UnifiedChatDataset returns tensors, just stack them.
        if not features:
            return {}
        keys = features[0].keys()
        for k in keys:
            batch[k] = torch.stack([f[k] for f in features])
        return batch

    # 8. Trainer
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
            max_prompt_length=max_length
        )
        trainer.add_callback(gen_cb)

    if cfg.get('data', {}).get('type') == 'xsum':
        from src.utils.evaluation import XSumGenerationCallback
        gen_cb = XSumGenerationCallback(
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            trainer=trainer,
            num_samples=eval_num_samples,
            max_prompt_length=max_length
        )
        trainer.add_callback(gen_cb)

    if cfg.get('data', {}).get('type') == 'healthcaremagic':
        from src.utils.evaluation import HealthCareMagicGenerationCallback
        gen_cb = HealthCareMagicGenerationCallback(
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            trainer=trainer,
            num_samples=eval_num_samples,
            max_prompt_length=max_length  # pass configured max_length
        )
        trainer.add_callback(gen_cb)

    if cfg.get('data', {}).get('type') == 'cuadqa':
        from src.utils.evaluation import CUADQAGenerationCallback
        gen_cb = CUADQAGenerationCallback(
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            trainer=trainer,
            num_samples=eval_num_samples,
            max_prompt_length=max_length  # pass configured max_length
        )
        trainer.add_callback(gen_cb)

    if cfg.get('data', {}).get('type') == 'magicoder':
        from src.utils.evaluation import MagicoderGenerationCallback
        gen_cb = MagicoderGenerationCallback(
            eval_dataset=eval_dataset, 
            tokenizer=tokenizer,
            trainer=trainer,
            num_samples=eval_num_samples,
            max_prompt_length=max_length  # pass configured max_length
        )
        trainer.add_callback(gen_cb)

    if training_args.gradient_checkpointing:
        model.config.use_cache = False
        if hasattr(model, 'enable_input_require_grads'):
            model.enable_input_require_grads()

    trainer.train()
    
    # [MODIFIED] Save into data_type subdir (final_model under out_dir)
    final_save_path = os.path.join(out_dir, "final_model")
    if _is_main():
        print(f"[INFO] Saving final model to {final_save_path}")
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)

if __name__ == '__main__':
    main()
