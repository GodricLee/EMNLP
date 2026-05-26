import torch
from transformers import TrainerCallback
try:
    import evaluate
except ImportError:
    evaluate = None
from tqdm import tqdm
import numpy as np
import sys

class AESLCGenerationCallback(TrainerCallback):
    """
    Generation evaluation callback for AESLC tasks.
    Uses `model.generate()` (beam search) instead of logits-based evaluation.
    """
    def __init__(self, eval_dataset, tokenizer, trainer=None, num_samples=50, max_prompt_length=1024):
        if evaluate is None:
            raise ImportError("The 'evaluate' library is missing. Please install it via `pip install evaluate rouge_score bert_score`.")
            
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.eval_samples = []
        self.max_prompt_length = max_prompt_length
        
        # Prefer `raw_samples` (raw text) over `samples` when available.
        # `raw_samples` should come from the Adapter and match training prompts.
        if hasattr(eval_dataset, 'raw_samples'):
            self.eval_samples = eval_dataset.raw_samples[:num_samples]
        elif hasattr(eval_dataset, 'samples'):
            # NOTE: If `dataset.samples` is already tokenized tensors, later code may fail.
            # We assume the dataset has been fixed to include raw (text) samples.
            self.eval_samples = eval_dataset.samples[:num_samples]
        else:
            print("Warning: eval_dataset does not have 'raw_samples' or 'samples' attribute. Skipping generation eval.")
        
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        # Run only on the main process
        if not state.is_local_process_zero:
            return
            
        if not self.eval_samples:
            return

        # Verify sample format to avoid KeyError
        if not isinstance(self.eval_samples[0], dict) or 'role_user' not in self.eval_samples[0]:
            print("[AESLC Eval] Error: eval_samples does not contain 'role_user'. Skipping generation.")
            return

        # Ensure we have the active training model
        if model is None and self.trainer is not None:
            model = self.trainer.model
        
        if model is None:
            print("[AESLC Eval] Warning: No model found for evaluation.")
            return

        print(f"\n[AESLC Eval] Starting Beam Search Generation on {len(self.eval_samples)} samples (Step {state.global_step})...")
        print(f"[AESLC Eval] Model training mode before eval: {model.training} (Should be True if called during training)")
        
        # 1. Unwrap model and force eval mode
        base_model = model
        # Unwrap DDP
        if hasattr(base_model, 'module'):
            base_model = base_model.module
        # Unwrap PEFT
        if hasattr(base_model, 'get_base_model'):
            base_model = base_model.get_base_model()
        
        # Force eval mode recursively to disable dropout and custom training logic
        model.eval()
        base_model.eval()
            
        # Temporarily allow logits output if the model forbids it during eval
        # so `generate()` can run without errors.
        saved_flag = getattr(base_model, 'return_loss_only_eval', None)
        if saved_flag is True:
            base_model.return_loss_only_eval = False
        
        preds = []
        refs = []
        
        device = model.device

        try:
            for idx, item in enumerate(tqdm(self.eval_samples, desc="Generating")):
                # 1. Get prompt ('role_user') and reference ('role_assistant').
                # [CONFIRMATION] Prompt is taken directly from Adapter without modification.
                prompt = item['role_user']
                label = item['role_assistant']
                
                # 2. Encode prompt (apply chat template if available)
                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                    messages = [{"role": "user", "content": prompt}]
                    input_ids = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=True, 
                        add_generation_prompt=True, 
                        return_tensors="pt"
                    )
                else:
                    print("[AESLC Eval] CRITICAL WARN: No chat template found! Generation will be garbage.")
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

                # 2.1 Length check/truncation to avoid OOM or training/inference mismatches
                if input_ids.shape[1] > self.max_prompt_length:
                    if idx < 3:
                        print(f"[AESLC Eval] Sample {idx} length {input_ids.shape[1]} > {self.max_prompt_length}. Keeping full length (truncation risky for chat templates).")
                
                input_ids = input_ids.to(device)
                attention_mask = torch.ones_like(input_ids).to(device)
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

                # Debug Logging for first 3 samples
                if idx < 3:
                    print(f"\n[AESLC DEBUG {idx}]")
                    # [CONFIRMATION] Print raw prompt (for debugging/verification)
                    print(f"RAW PROMPT (from adapter): {repr(prompt[:100])}...")
                    decoded_input = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
                    # Print prompt tail to confirm ending tokens
                    print(f"INPUT TAIL (Tokenized): ...{repr(decoded_input[-200:])}")
                    sys.stdout.flush()
                
                # 3. Run Beam Search generation
                with torch.no_grad():
                    # Compatibility with DDP/PEFT model wrappers
                    gen_model = model.module if hasattr(model, 'module') else model
                    
                    # Define terminators for Llama 3 to prevent infinite loops
                    terminators = [
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    outputs = gen_model.generate(
                        **inputs,
                        min_length=1,
                        max_new_tokens=10,       # Reduced to prevent long hallucinations
                        num_beams=8,
                        early_stopping=True,
                        no_repeat_ngram_size=3,
                        repetition_penalty=2.0,  # Strictly forbid loops
                        length_penalty=-2.0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=terminators # Explicitly pass Llama 3 stop tokens
                    )
                
                # 4. Decode output and remove the prompt portion (outputs include input_ids)
                input_len = inputs['input_ids'].shape[1]
                generated_ids = outputs[0][input_len:]
                decoded_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                # Debug Logging Output
                if idx < 3:
                    print(f"GENERATED: {repr(decoded_output)}")
                    print(f"REFERENCE: {repr(label)}")
                    sys.stdout.flush()

                preds.append(decoded_output)
                refs.append(label)
        finally:
            # Restore flag to avoid affecting subsequent eval loss calculation
            if saved_flag is True:
                base_model.return_loss_only_eval = True

        # 5. Compute metrics
        print("[AESLC Eval] Computing ROUGE & BERTScore...")
        try:
            rouge_res = self.rouge.compute(predictions=preds, references=refs)
            bert_res = self.bertscore.compute(predictions=preds, references=refs, lang="en")
            
            # 6. Print results
            print("\n" + "="*40)
            print(f"📊 Generation Metrics (Step {state.global_step})")
            print(f"ROUGE-1: {rouge_res['rouge1']:.4f}")
            print(f"ROUGE-2: {rouge_res['rouge2']:.4f}")
            print(f"ROUGE-L: {rouge_res['rougeL']:.4f}")
            print(f"BERTScore F1: {np.mean(bert_res['f1']):.4f}")
            print("="*40 + "\n")

            computed_metrics = {
                "eval_rouge1": rouge_res['rouge1'],
                "eval_rougeL": rouge_res['rougeL'],
                "eval_bertscore_f1": np.mean(bert_res['f1'])
            }
            
            # Use `trainer.log()` to record metrics (handles TB/WandB/console)
            if self.trainer:
                self.trainer.log(computed_metrics)
                
            # Also update the Trainer `metrics` dict so best-metric logic can pick up eval_rougeL
            if metrics is not None:
                metrics.update(computed_metrics)
        except Exception as e:
            print(f"[AESLC Eval] Metric computation failed: {e}")


class XSumGenerationCallback(TrainerCallback):
    """
    Generation evaluation callback for XSum tasks.
    Uses `model.generate()` (beam search) instead of logits-based evaluation.
    """
    def __init__(self, eval_dataset, tokenizer, trainer=None, num_samples=50, max_prompt_length=1024):
        if evaluate is None:
            raise ImportError("The 'evaluate' library is missing. Please install it via `pip install evaluate rouge_score bert_score`.")
            
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.eval_samples = []
        self.max_prompt_length = max_prompt_length
        
        # Prefer `raw_samples` (raw text) over `samples` when available.
        # `raw_samples` should come from the Adapter and match training prompts.
        if hasattr(eval_dataset, 'raw_samples'):
            self.eval_samples = eval_dataset.raw_samples[:num_samples]
        elif hasattr(eval_dataset, 'samples'):
            # NOTE: If `dataset.samples` is already tokenized tensors, later code may fail.
            # We assume the dataset has been fixed to include raw (text) samples.
            self.eval_samples = eval_dataset.samples[:num_samples]
        else:
            print("Warning: eval_dataset does not have 'raw_samples' or 'samples' attribute. Skipping generation eval.")
        
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        # Run only on the main process
        if not state.is_local_process_zero:
            return
            
        if not self.eval_samples:
            return

        # Verify sample format to avoid KeyError
        if not isinstance(self.eval_samples[0], dict) or 'role_user' not in self.eval_samples[0]:
            print("[XSum Eval] Error: eval_samples does not contain 'role_user'. Skipping generation.")
            return

        # Ensure we have the active training model
        if model is None and self.trainer is not None:
            model = self.trainer.model
        
        if model is None:
            print("[XSum Eval] Warning: No model found for evaluation.")
            return

        print(f"\n[XSum Eval] Starting Beam Search Generation on {len(self.eval_samples)} samples (Step {state.global_step})...")
        print(f"[XSum Eval] Model training mode before eval: {model.training} (Should be True if called during training)")
        
        # 1. Unwrap model and force eval mode
        base_model = model
        # Unwrap DDP
        if hasattr(base_model, 'module'):
            base_model = base_model.module
        # Unwrap PEFT
        if hasattr(base_model, 'get_base_model'):
            base_model = base_model.get_base_model()
        
        # Force eval mode recursively to disable dropout and custom training logic
        model.eval()
        base_model.eval()
            
        # Temporarily allow logits output if the model forbids it during eval
        # so `generate()` can run without errors.
        saved_flag = getattr(base_model, 'return_loss_only_eval', None)
        if saved_flag is True:
            base_model.return_loss_only_eval = False
        
        preds = []
        refs = []
        
        device = model.device

        try:
            for idx, item in enumerate(tqdm(self.eval_samples, desc="Generating")):
                # 1. Get prompt ('role_user') and reference ('role_assistant').
                # [CONFIRMATION] Prompt is taken directly from Adapter without modification.
                prompt = item['role_user']
                label = item['role_assistant']
                
                # 2. Encode prompt (apply chat template if available)
                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                    messages = [{"role": "user", "content": prompt}]
                    input_ids = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=True, 
                        add_generation_prompt=True, 
                        return_tensors="pt"
                    )
                else:
                    print("[XSum Eval] CRITICAL WARN: No chat template found! Generation will be garbage.")
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

                # 2.1 Length check/truncation to avoid OOM or training/inference mismatches
                if input_ids.shape[1] > self.max_prompt_length:
                    if idx < 3:
                        print(f"[XSum Eval] Sample {idx} length {input_ids.shape[1]} > {self.max_prompt_length}. Keeping full length (truncation risky for chat templates).")
                
                input_ids = input_ids.to(device)
                attention_mask = torch.ones_like(input_ids).to(device)
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

                # Debug Logging for first 3 samples
                if idx < 3:
                    print(f"\n[XSum DEBUG {idx}]")
                    # [CONFIRMATION] Print raw prompt (for debugging/verification)
                    print(f"RAW PROMPT (from adapter): {repr(prompt[:100])}...") 
                    decoded_input = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
                    # Print prompt tail to confirm ending tokens
                    print(f"INPUT TAIL (Tokenized): ...{repr(decoded_input[-200:])}")
                    sys.stdout.flush()
                
                # 3. Run Beam Search generation
                with torch.no_grad():
                    # Compatibility with DDP/PEFT model wrappers
                    gen_model = model.module if hasattr(model, 'module') else model
                    
                    # Define terminators for Llama 3 to prevent infinite loops
                    terminators = [
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    outputs = gen_model.generate(
                        **inputs,
                        # 1. Length control
                        max_new_tokens=32,    # 32 tokens validated for this task
                        min_new_tokens=3,     # avoid empty or single-token outputs

                        # 2. Length penalty to favor shorter outputs
                        length_penalty=0.8,   # <1.0 encourages shorter sequences

                        # 3. Search strategy
                        num_beams=8,
                        early_stopping=True,

                        # 4. Repetition prevention
                        repetition_penalty=1.2,

                        # 5. Stop token configuration
                        pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                        eos_token_id=terminators 
                    )
                
                # 4. Decode output and remove the prompt portion (outputs include input_ids)
                input_len = inputs['input_ids'].shape[1]
                generated_ids = outputs[0][input_len:]
                decoded_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                # Debug Logging Output
                if idx < 3:
                    print(f"GENERATED: {repr(decoded_output)}")
                    print(f"REFERENCE: {repr(label)}")
                    sys.stdout.flush()

                preds.append(decoded_output)
                refs.append(label)
        finally:
            # Restore flag to avoid affecting subsequent eval loss calculation
            if saved_flag is True:
                base_model.return_loss_only_eval = True

        # 5. Compute metrics
        print("[XSum Eval] Computing ROUGE & BERTScore...")
        try:
            rouge_res = self.rouge.compute(predictions=preds, references=refs)
            bert_res = self.bertscore.compute(predictions=preds, references=refs, lang="en")
            
            # 6. Print results
            print("\n" + "="*40)
            print(f"📊 Generation Metrics (Step {state.global_step})")
            print(f"ROUGE-1: {rouge_res['rouge1']:.4f}")
            print(f"ROUGE-2: {rouge_res['rouge2']:.4f}")
            print(f"ROUGE-L: {rouge_res['rougeL']:.4f}")
            print(f"BERTScore F1: {np.mean(bert_res['f1']):.4f}")
            print("="*40 + "\n")

            computed_metrics = {
                "eval_rouge1": rouge_res['rouge1'],
                "eval_rougeL": rouge_res['rougeL'],
                "eval_bertscore_f1": np.mean(bert_res['f1'])
            }
            
            # Use `trainer.log()` to record metrics (handles TB/WandB/console)
            if self.trainer:
                self.trainer.log(computed_metrics)
                
            # Also update the Trainer `metrics` dict so best-metric logic can pick up eval_rougeL
            if metrics is not None:
                metrics.update(computed_metrics)
        except Exception as e:
            print(f"[XSum Eval] Metric computation failed: {e}")


class HealthCareMagicGenerationCallback(TrainerCallback):
    """
    Generation evaluation callback for HealthCareMagic tasks.
    Uses `model.generate()` (beam search) instead of logits-based evaluation.
    """
    def __init__(self, eval_dataset, tokenizer, trainer=None, num_samples=50, max_prompt_length=1024):
        if evaluate is None:
            raise ImportError("The 'evaluate' library is missing. Please install it via `pip install evaluate rouge_score bert_score`.")
            
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.eval_samples = []
        self.max_prompt_length = max_prompt_length
        
        # Prefer `raw_samples` (raw text) over `samples` when available.
        # `raw_samples` should come from the Adapter and match training prompts.
        if hasattr(eval_dataset, 'raw_samples'):
            self.eval_samples = eval_dataset.raw_samples[:num_samples]
        elif hasattr(eval_dataset, 'samples'):
            # NOTE: If `dataset.samples` is already tokenized tensors, later code may fail.
            # We assume the dataset has been fixed to include raw (text) samples.
            self.eval_samples = eval_dataset.samples[:num_samples]
        else:
            print("Warning: eval_dataset does not have 'raw_samples' or 'samples' attribute. Skipping generation eval.")
        
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        # Run only on the main process
        if not state.is_local_process_zero:
            return
            
        if not self.eval_samples:
            return

        # Verify sample format to avoid KeyError
        if not isinstance(self.eval_samples[0], dict) or 'role_user' not in self.eval_samples[0]:
            print("[HealthCareMagic Eval] Error: eval_samples does not contain 'role_user'. Skipping generation.")
            return

        # Ensure we have the active training model
        if model is None and self.trainer is not None:
            model = self.trainer.model
        
        if model is None:
            print("[HealthCareMagic Eval] Warning: No model found for evaluation.")
            return

        print(f"\n[HealthCareMagic Eval] Starting Beam Search Generation on {len(self.eval_samples)} samples (Step {state.global_step})...")
        print(f"[HealthCareMagic Eval] Model training mode before eval: {model.training} (Should be True if called during training)")
        
        # 1. Unwrap model and force eval mode
        base_model = model
        # Unwrap DDP
        if hasattr(base_model, 'module'):
            base_model = base_model.module
        # Unwrap PEFT
        if hasattr(base_model, 'get_base_model'):
            base_model = base_model.get_base_model()
        
        # Force eval mode recursively to disable dropout and custom training logic
        model.eval()
        base_model.eval()
            
        # Temporarily allow logits output if the model forbids it during eval
        # so `generate()` can run without errors.
        saved_flag = getattr(base_model, 'return_loss_only_eval', None)
        if saved_flag is True:
            base_model.return_loss_only_eval = False
        
        preds = []
        refs = []
        
        device = model.device

        try:
            for idx, item in enumerate(tqdm(self.eval_samples, desc="Generating")):
                # 1. Get prompt ('role_user') and reference ('role_assistant').
                # [CONFIRMATION] Prompt is taken directly from Adapter without modification.
                prompt = item['role_user']
                label = item['role_assistant']
                
                # 2. Encode prompt (apply chat template if available)
                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                    messages = [{"role": "user", "content": prompt}]
                    input_ids = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=True, 
                        add_generation_prompt=True, 
                        return_tensors="pt"
                    )
                else:
                    print("[HealthCareMagic Eval] CRITICAL WARN: No chat template found! Generation will be garbage.")
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

                # 2.1 Length check/truncation to avoid OOM or training/inference mismatches
                if input_ids.shape[1] > self.max_prompt_length:
                    if idx < 3:
                        print(f"[HealthCareMagic Eval] Sample {idx} length {input_ids.shape[1]} > {self.max_prompt_length}. Keeping full length (truncation risky for chat templates).")
                
                input_ids = input_ids.to(device)
                attention_mask = torch.ones_like(input_ids).to(device)
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

                # Debug Logging for first 3 samples
                if idx < 3:
                    print(f"\n[HealthCareMagic DEBUG {idx}]")
                    # [CONFIRMATION] Print raw prompt (for debugging/verification)
                    print(f"RAW PROMPT (from adapter): {repr(prompt[:100])}...") 
                    decoded_input = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
                    # Print prompt tail to confirm ending tokens
                    print(f"INPUT TAIL (Tokenized): ...{repr(decoded_input[-200:])}")
                    sys.stdout.flush()
                
                # 3. Run Beam Search generation
                with torch.no_grad():
                    # Compatibility with DDP/PEFT model wrappers
                    gen_model = model.module if hasattr(model, 'module') else model
                    
                    # Define terminators for Llama 3 to prevent infinite loops
                    terminators = [
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    outputs = gen_model.generate(
                        **inputs,
                        # 1. Length control
                        max_new_tokens=32,
                        min_new_tokens=3,

                        # 2. Length penalty to favor shorter outputs
                        length_penalty=0.8,

                        # 3. Search strategy
                        num_beams=8,
                        early_stopping=True,

                        # 4. Repetition prevention
                        repetition_penalty=1.2,

                        # 5. Stop token configuration
                        pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                        eos_token_id=terminators 
                    )
                
                # 4. Decode output and remove the prompt portion (outputs include input_ids)
                input_len = inputs['input_ids'].shape[1]
                generated_ids = outputs[0][input_len:]
                decoded_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                # Debug Logging Output
                if idx < 3:
                    print(f"GENERATED: {repr(decoded_output)}")
                    print(f"REFERENCE: {repr(label)}")
                    sys.stdout.flush()

                preds.append(decoded_output)
                refs.append(label)
        finally:
            # Restore flag to avoid affecting subsequent eval loss calculation
            if saved_flag is True:
                base_model.return_loss_only_eval = True

        # 5. Compute metrics
        print("[HealthCareMagic Eval] Computing ROUGE & BERTScore...")
        try:
            rouge_res = self.rouge.compute(predictions=preds, references=refs)
            bert_res = self.bertscore.compute(predictions=preds, references=refs, lang="en")
            
            # 6. Print results
            print("\n" + "="*40)
            print(f"📊 Generation Metrics (Step {state.global_step})")
            print(f"ROUGE-1: {rouge_res['rouge1']:.4f}")
            print(f"ROUGE-2: {rouge_res['rouge2']:.4f}")
            print(f"ROUGE-L: {rouge_res['rougeL']:.4f}")
            print(f"BERTScore F1: {np.mean(bert_res['f1']):.4f}")
            print("="*40 + "\n")

            computed_metrics = {
                "eval_rouge1": rouge_res['rouge1'],
                "eval_rougeL": rouge_res['rougeL'],
                "eval_bertscore_f1": np.mean(bert_res['f1'])
            }
            
            # Use `trainer.log()` to record metrics (handles TB/WandB/console)
            if self.trainer:
                self.trainer.log(computed_metrics)
                
            # Also update the Trainer `metrics` dict so best-metric logic can pick up eval_rougeL
            if metrics is not None:
                metrics.update(computed_metrics)
        except Exception as e:
            print(f"[HealthCareMagic Eval] Metric computation failed: {e}")
            
class CUADQAGenerationCallback(TrainerCallback):
    """
    Generation evaluation callback for CUAD QA tasks.
    Uses `model.generate()` (beam search) instead of logits-based evaluation.
    """
    def __init__(self, eval_dataset, tokenizer, trainer=None, num_samples=50, max_prompt_length=1024):
        if evaluate is None:
            raise ImportError("The 'evaluate' library is missing. Please install it via `pip install evaluate rouge_score bert_score`.")
            
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.eval_samples = []
        self.max_prompt_length = max_prompt_length
        
        # Prefer `raw_samples` (raw text) over `samples` when available.
        # `raw_samples` should come from the Adapter and match training prompts.
        if hasattr(eval_dataset, 'raw_samples'):
            self.eval_samples = eval_dataset.raw_samples[:num_samples]
        elif hasattr(eval_dataset, 'samples'):
            # NOTE: If `dataset.samples` is already tokenized tensors, later code may fail.
            # We assume the dataset has been fixed to include raw (text) samples.
            self.eval_samples = eval_dataset.samples[:num_samples]
        else:
            print("Warning: eval_dataset does not have 'raw_samples' or 'samples' attribute. Skipping generation eval.")
        
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        # Run only on the main process
        if not state.is_local_process_zero:
            return
            
        if not self.eval_samples:
            return

        # Verify sample format to avoid KeyError
        if not isinstance(self.eval_samples[0], dict) or 'role_user' not in self.eval_samples[0]:
            print("[CUAD QA Eval] Error: eval_samples does not contain 'role_user'. Skipping generation.")
            return

        # Ensure we have the active training model
        if model is None and self.trainer is not None:
            model = self.trainer.model
        
        if model is None:
            print("[CUAD QA Eval] Warning: No model found for evaluation.")
            return

        print(f"\n[CUAD QA Eval] Starting Beam Search Generation on {len(self.eval_samples)} samples (Step {state.global_step})...")
        print(f"[CUAD QA Eval] Model training mode before eval: {model.training} (Should be True if called during training)")
        
        # 1. Unwrap model and force eval mode
        base_model = model
        # Unwrap DDP
        if hasattr(base_model, 'module'):
            base_model = base_model.module
        # Unwrap PEFT
        if hasattr(base_model, 'get_base_model'):
            base_model = base_model.get_base_model()
        
        # Force eval mode recursively to disable dropout and custom training logic
        model.eval()
        base_model.eval()
            
        # Temporarily allow logits output if the model forbids it during eval
        # so `generate()` can run without errors.
        saved_flag = getattr(base_model, 'return_loss_only_eval', None)
        if saved_flag is True:
            base_model.return_loss_only_eval = False
        
        preds = []
        refs = []
        
        device = model.device

        try:
            for idx, item in enumerate(tqdm(self.eval_samples, desc="Generating")):
                # 1. Get prompt ('role_user') and reference ('role_assistant').
                # [CONFIRMATION] Prompt is taken directly from Adapter without modification.
                prompt = item['role_user']
                label = item['role_assistant']
                
                # 2. Encode prompt (apply chat template if available)
                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                    messages = [{"role": "user", "content": prompt}]
                    input_ids = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=True, 
                        add_generation_prompt=True, 
                        return_tensors="pt"
                    )
                else:
                    print("[CUAD QA Eval] CRITICAL WARN: No chat template found! Generation will be garbage.")
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

                # 2.1 Length check/truncation to avoid OOM or training/inference mismatches
                if input_ids.shape[1] > self.max_prompt_length:
                    if idx < 3:
                        print(f"[CUAD QA Eval] Sample {idx} length {input_ids.shape[1]} > {self.max_prompt_length}. Keeping full length (truncation risky for chat templates).")
                
                input_ids = input_ids.to(device)
                attention_mask = torch.ones_like(input_ids).to(device)
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

                # Debug Logging for first 3 samples
                if idx < 3:
                    print(f"\n[CUAD QA DEBUG {idx}]")
                    # [CONFIRMATION] Print raw prompt (for debugging/verification)
                    print(f"RAW PROMPT (from adapter): {repr(prompt[:100])}...") 
                    decoded_input = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
                    # Print prompt tail to confirm ending tokens
                    print(f"INPUT TAIL (Tokenized): ...{repr(decoded_input[-200:])}")
                    sys.stdout.flush()
                
                # 3. Run Beam Search generation
                with torch.no_grad():
                    # Compatibility with DDP/PEFT model wrappers
                    gen_model = model.module if hasattr(model, 'module') else model
                    
                    # Define terminators for Llama 3 to prevent infinite loops
                    terminators = [
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    outputs = gen_model.generate(
                        **inputs,
                        # 1. Length control
                        max_new_tokens=32,
                        min_new_tokens=3,

                        # 2. Length penalty to favor shorter outputs
                        length_penalty=0.8,

                        # 3. Search strategy
                        num_beams=8,
                        early_stopping=True,

                        # 4. Repetition prevention
                        repetition_penalty=1.2,

                        # 5. Stop token configuration
                        pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                        eos_token_id=terminators 
                    )
                
                # 4. Decode output and remove the prompt portion (outputs include input_ids)
                input_len = inputs['input_ids'].shape[1]
                generated_ids = outputs[0][input_len:]
                decoded_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                # Debug Logging Output
                if idx < 3:
                    print(f"GENERATED: {repr(decoded_output)}")
                    print(f"REFERENCE: {repr(label)}")
                    sys.stdout.flush()

                preds.append(decoded_output)
                refs.append(label)
        finally:
            # Restore flag to avoid affecting subsequent eval loss calculation
            if saved_flag is True:
                base_model.return_loss_only_eval = True

        # 5. Compute metrics
        print("[CUAD QA Eval] Computing ROUGE & BERTScore...")
        try:
            rouge_res = self.rouge.compute(predictions=preds, references=refs)
            bert_res = self.bertscore.compute(predictions=preds, references=refs, lang="en")
            
            # 6. Print results
            print("\n" + "="*40)
            print(f"📊 Generation Metrics (Step {state.global_step})")
            print(f"ROUGE-1: {rouge_res['rouge1']:.4f}")
            print(f"ROUGE-2: {rouge_res['rouge2']:.4f}")
            print(f"ROUGE-L: {rouge_res['rougeL']:.4f}")
            print(f"BERTScore F1: {np.mean(bert_res['f1']):.4f}")
            print("="*40 + "\n")

            computed_metrics = {
                "eval_rouge1": rouge_res['rouge1'],
                "eval_rougeL": rouge_res['rougeL'],
                "eval_bertscore_f1": np.mean(bert_res['f1'])
            }
            
            # Use `trainer.log()` to record metrics (handles TB/WandB/console)
            if self.trainer:
                self.trainer.log(computed_metrics)
                
            # Also update the Trainer `metrics` dict so best-metric logic can pick up eval_rougeL
            if metrics is not None:
                metrics.update(computed_metrics)
        except Exception as e:
            print(f"[CUAD QA Eval] Metric computation failed: {e}")

class MagicoderGenerationCallback(TrainerCallback):
    """
    Generation evaluation callback for Magicoder tasks.
    Uses `model.generate()` (beam search) instead of logits-based evaluation.
    """
    def __init__(self, eval_dataset, tokenizer, trainer=None, num_samples=50, max_prompt_length=1024):
        if evaluate is None:
            raise ImportError("The 'evaluate' library is missing. Please install it via `pip install evaluate rouge_score bert_score`.")
            
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.eval_samples = []
        self.max_prompt_length = max_prompt_length
        
        # Prefer `raw_samples` (raw text) over `samples` when available.
        # `raw_samples` should come from the Adapter and match training prompts.
        if hasattr(eval_dataset, 'raw_samples'):
            self.eval_samples = eval_dataset.raw_samples[:num_samples]
        elif hasattr(eval_dataset, 'samples'):
            # NOTE: If `dataset.samples` is already tokenized tensors, later code may fail.
            # We assume the dataset has been fixed to include raw (text) samples.
            self.eval_samples = eval_dataset.samples[:num_samples]
        else:
            print("Warning: eval_dataset does not have 'raw_samples' or 'samples' attribute. Skipping generation eval.")
        
        self.rouge = evaluate.load("rouge")
        self.bertscore = evaluate.load("bertscore")

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        # Run only on the main process
        if not state.is_local_process_zero:
            return
            
        if not self.eval_samples:
            return

        # Verify sample format to avoid KeyError
        if not isinstance(self.eval_samples[0], dict) or 'role_user' not in self.eval_samples[0]:
            print("[CUAD QA Eval] Error: eval_samples does not contain 'role_user'. Skipping generation.")
            return

        # Ensure we have the active training model
        if model is None and self.trainer is not None:
            model = self.trainer.model
        
        if model is None:
            print("[Magicoder Eval] Warning: No model found for evaluation.")
            return

        print(f"\n[Magicoder Eval] Starting Beam Search Generation on {len(self.eval_samples)} samples (Step {state.global_step})...")
        print(f"[Magicoder Eval] Model training mode before eval: {model.training} (Should be True if called during training)")
        
        # 1. Unwrap model and force eval mode
        base_model = model
        # Unwrap DDP
        if hasattr(base_model, 'module'):
            base_model = base_model.module
        # Unwrap PEFT
        if hasattr(base_model, 'get_base_model'):
            base_model = base_model.get_base_model()
        
        # Force eval mode recursively to disable dropout and custom training logic
        model.eval()
        base_model.eval()
            
        # Temporarily allow logits output if the model forbids it during eval
        # so `generate()` can run without errors.
        saved_flag = getattr(base_model, 'return_loss_only_eval', None)
        if saved_flag is True:
            base_model.return_loss_only_eval = False
        
        preds = []
        refs = []
        
        device = model.device

        try:
            for idx, item in enumerate(tqdm(self.eval_samples, desc="Generating")):
                # 1. Get prompt ('role_user') and reference ('role_assistant').
                # [CONFIRMATION] Prompt is taken directly from Adapter without modification.
                prompt = item['role_user']
                label = item['role_assistant']
                
                # 2. Encode prompt (apply chat template if available)
                if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
                    messages = [{"role": "user", "content": prompt}]
                    input_ids = self.tokenizer.apply_chat_template(
                        messages, 
                        tokenize=True, 
                        add_generation_prompt=True, 
                        return_tensors="pt"
                    )
                else:
                    print("[Magicoder Eval] CRITICAL WARN: No chat template found! Generation will be garbage.")
                    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

                # 2.1 Length check/truncation to avoid OOM or training/inference mismatches
                if input_ids.shape[1] > self.max_prompt_length:
                    if idx < 3:
                        print(f"[Magicoder Eval] Sample {idx} length {input_ids.shape[1]} > {self.max_prompt_length}. Keeping full length (truncation risky for chat templates).")
                
                input_ids = input_ids.to(device)
                attention_mask = torch.ones_like(input_ids).to(device)
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}

                # Debug Logging for first 3 samples
                if idx < 3:
                    print(f"\n[Magicoder DEBUG {idx}]")
                    # [CONFIRMATION] Print raw prompt (for debugging/verification)
                    print(f"RAW PROMPT (from adapter): {repr(prompt[:100])}...") 
                    decoded_input = self.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False)
                    # Print prompt tail to confirm ending tokens
                    print(f"INPUT TAIL (Tokenized): ...{repr(decoded_input[-200:])}")
                    sys.stdout.flush()
                
                # 3. Run Beam Search generation
                with torch.no_grad():
                    gen_model = model.module if hasattr(model, 'module') else model
                    
                    # Define terminators for Llama 3 to prevent infinite loops
                    terminators = [
                        self.tokenizer.eos_token_id,
                        self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
                    ]

                    outputs = gen_model.generate(
                        **inputs,
                        # 1. Length control
                        max_new_tokens=32,
                        min_new_tokens=3,

                        # 2. Length penalty to favor shorter outputs
                        length_penalty=1,

                        # 3. Search strategy
                        num_beams=8,
                        early_stopping=True,

                        # 4. Repetition prevention
                        repetition_penalty=1.2,

                        # 5. Stop token configuration
                        pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id,
                        eos_token_id=terminators 
                    )
                
                # 4. Decode output and remove the prompt portion (outputs include input_ids)
                input_len = inputs['input_ids'].shape[1]
                generated_ids = outputs[0][input_len:]
                decoded_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                # Debug Logging Output
                if idx < 3:
                    print(f"GENERATED: {repr(decoded_output)}")
                    print(f"REFERENCE: {repr(label)}")
                    sys.stdout.flush()

                preds.append(decoded_output)
                refs.append(label)
        finally:
            # Restore flag to avoid affecting subsequent eval loss calculation
            if saved_flag is True:
                base_model.return_loss_only_eval = True

        # 5. Compute metrics
        print("[Magicoder Eval] Computing ROUGE & BERTScore...")
        try:
            rouge_res = self.rouge.compute(predictions=preds, references=refs)
            bert_res = self.bertscore.compute(predictions=preds, references=refs, lang="en")
            
            # 6. Print results
            print("\n" + "="*40)
            print(f"📊 Generation Metrics (Step {state.global_step})")
            print(f"ROUGE-1: {rouge_res['rouge1']:.4f}")
            print(f"ROUGE-2: {rouge_res['rouge2']:.4f}")
            print(f"ROUGE-L: {rouge_res['rougeL']:.4f}")
            print(f"BERTScore F1: {np.mean(bert_res['f1']):.4f}")
            print("="*40 + "\n")

            computed_metrics = {
                "eval_rouge1": rouge_res['rouge1'],
                "eval_rougeL": rouge_res['rougeL'],
                "eval_bertscore_f1": np.mean(bert_res['f1'])
            }
            
            # Use `trainer.log()` to record metrics (handles TB/WandB/console)
            if self.trainer:
                self.trainer.log(computed_metrics)
                
            # Also update the Trainer `metrics` dict so best-metric logic can pick up eval_rougeL
            if metrics is not None:
                metrics.update(computed_metrics)
        except Exception as e:
            print(f"[Magicoder Eval] Metric computation failed: {e}")