import sys
import os
import argparse
import torch
import json
import time
import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate

# Ensure src module is visible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.adapters import AESLCAdapter
from src.data.dataset import UnifiedChatDataset
from src.models.modulated_llama import ModulatedLlamaForCausalLM

class InferenceDataset(Dataset):
    """
    Dataset wrapper for inference. 
    It uses the raw samples from UnifiedChatDataset to ensure consistency,
    but tokenizes only the user prompt for generation.
    """
    def __init__(self, raw_samples, tokenizer):
        self.samples = raw_samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = item['role_user']
        reference = item['role_assistant']
        
        # RESTORED: Strictly mirror src/utils/evaluation.py logic
        # We MUST use apply_chat_template because the model (Llama-3-Instruct) 
        # relies on the specific system prompts and header tokens injected by it.
        
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors='pt'
            )
        else:
            # Fallback only if template is missing
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        
        # Remove batch dimension [1, L] -> [L]
        input_ids = input_ids.squeeze(0)
        
        return {
            "input_ids": input_ids,
            "reference": reference,
            "prompt_text": prompt
        }

class DataCollatorForGeneration:
    """
    Collator that handles left-padding for Causal LM generation.
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, batch):
        # Extract inputs
        input_ids_list = [item['input_ids'] for item in batch]
        references = [item['reference'] for item in batch]
        prompts = [item['prompt_text'] for item in batch]
        
        # Calculate max length in this batch
        max_len = max(len(ids) for ids in input_ids_list)
        
        # Prepare tensors with left padding
        batch_size = len(input_ids_list)
        padded_input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        for i, ids in enumerate(input_ids_list):
            seq_len = len(ids)
            # Left padding: place sequence at the end
            padded_input_ids[i, -seq_len:] = ids
            attention_mask[i, -seq_len:] = 1
            
        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "references": references,
            "prompts": prompts
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Full AESLC Evaluation Script")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved LoRA checkpoint (e.g., output/date/checkpoint-xxx)")
    parser.add_argument("--base_model_name", type=str, default=None, help="Path to base model")
    parser.add_argument("--data_file", type=str, default="src/data/aeslc/aeslc_clean_test.json", help="Path to test data")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for generation (Use 1 for strict debugging)")
    parser.add_argument("--output_file", type=str, default="temp", help="Path to save predictions")
    parser.add_argument("--device", type=str, default=None, help="Manually specify device (e.g. 'cuda:0', 'cpu'). If None, infers from env.")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples to evaluate (e.g. 50 for parity with training eval)")
    parser.add_argument("--use_flash_attn", action="store_true", help="Enable Flash Attention 2 for faster inference and lower memory usage")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Tokenizer & Model
    # IMPROVEMENT: Try loading tokenizer from adapter path first to ensure consistency with training
    try:
        print(f"[INFO] Attempting to load tokenizer from adapter path: {args.model_path}")
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    except Exception as e:
        print(f"[INFO] Fallback: Loading tokenizer from base model: {args.base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=True)

    tokenizer.padding_side = "left"  # Critical: Force left padding for batched generation
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    # FIX: Handle device placement manually to avoid DTensor/device_map="auto" issues in DDP
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    if args.device:
        device = torch.device(args.device)
        if device.type == 'cuda':
            torch.cuda.set_device(device)
    elif local_rank != -1:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] Using device: {device}")
    if device.type == 'cuda':
        try:
            print(f"[INFO] Device Name: {torch.cuda.get_device_name(device)}")
            print(f"[INFO] Note: If you used CUDA_VISIBLE_DEVICES=1, 'cuda:0' refers to that specific GPU.")
        except Exception:
            pass

    # CHANGED: Use ModulatedLlamaForCausalLM to match infer_modulated.py
    print(f"[INFO] Loading base (modulated) model from {args.base_model_name}")
    
    # [NEW] Flash Attention Support
    attn_impl = None
    if args.use_flash_attn:
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
            print(f"[INFO] Enabling Flash Attention 2 (version {flash_attn.__version__})")
        except ImportError:
            print("[WARN] Flash Attention 2 requested but not installed. Falling back to default.")
            
    model = ModulatedLlamaForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map=None, # Explicitly None as in infer_modulated.py
        attn_implementation=attn_impl
    )
    
    # Sync pad_token_id to model config (Consistency with training script)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # IMPROVEMENT: Resize embeddings if tokenizer has more tokens (e.g. special tokens added during training)
    # This is critical if the training script performed resizing.
    if model.get_input_embeddings().weight.size(0) < len(tokenizer):
        print(f"[INFO] Resizing token embeddings from {model.get_input_embeddings().weight.size(0)} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        
    print(f"[INFO] Loading LoRA adapter: {args.model_path}")
    # Load the adapter. Matches logic in infer_modulated.py
    if os.path.isdir(args.model_path) and any(n.startswith('adapter') for n in os.listdir(args.model_path)):
        model = PeftModel.from_pretrained(model, args.model_path)
        print(f"[INFO] LoRA Adapter Loaded. Active adapters: {model.active_adapters}")
    else:
        print(f"[WARN] No LoRA adapter found in {args.model_path}, using base weights only")
    
    model.to(device)
    model.eval()
    
    # Ensure generation config picks up the pad token
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    
    # 2. Load Data
    print(f"[INFO] Loading data from: {args.data_file}")
    # Mock config to reuse AESLCAdapter
    mock_config = {
        'data': {
            'type': 'aeslc',
            'aeslc': {
                'train_file': '', # Not used
                'eval_file': args.data_file
            }
        }
    }
    
    adapter = AESLCAdapter(mock_config)
    # AESLCAdapter uses 'eval_file' when split is not 'train'
    raw_data = adapter.load_data('test')
    
    if not raw_data:
        print("[ERROR] No data loaded. Check data path.")
        return

    # RESTORED: Use UnifiedChatDataset to ensure exact consistency with training pipeline logic and logs
    print("[INFO] Instantiating UnifiedChatDataset to match training pipeline...")
    # We use a default max_length and empty patterns as they don't affect the raw_samples content 
    # used for generation, but this triggers the [DATA-SAMPLE] logs the user expects.
    temp_dataset = UnifiedChatDataset(raw_data, tokenizer, max_length=2048)
    eval_samples = temp_dataset.raw_samples
    
    # Apply limit if specified
    if args.limit is not None and args.limit > 0:
        print(f"[INFO] Limiting evaluation to first {args.limit} samples (Training Eval usually uses 50).")
        eval_samples = eval_samples[:args.limit]
    
    print(f"[INFO] Total samples to evaluate: {len(eval_samples)}")
    
    # 3. Prepare DataLoader
    inference_dataset = InferenceDataset(eval_samples, tokenizer)
    collator = DataCollatorForGeneration(tokenizer)
    dataloader = DataLoader(
        inference_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collator,
        num_workers=4
    )
    
    # 4. Metrics
    print("[INFO] Loading metrics...")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    
    # 5. Generation Loop
    predictions = []
    references = []
    results_log = []
    
    # Use the device we determined earlier
    start_time = time.time()
    total_batches = len(dataloader)
    
    # Define terminators for Llama 3 to prevent infinite generation loops
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    print(f"[INFO] Starting generation (Batch Size: {args.batch_size})...")
    
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        batch_start = time.time()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        batch_refs = batch['references']
        batch_prompts = batch['prompts']
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                min_length=1,
                max_new_tokens=10,       # Reduced to prevent long hallucinations
                num_beams=8,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=2.0,  # Strictly forbid loops
                length_penalty=-2.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators # Explicitly pass Llama 3 stop tokens
            )
            
        # Decode
        # Slice off the input prompt
        input_len = input_ids.shape[1]
        generated_tokens = outputs[:, input_len:]
        decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        
        # Clean up predictions
        decoded_preds = [p.strip() for p in decoded_preds]
        
        predictions.extend(decoded_preds)
        references.extend(batch_refs)
        
        # Log results
        for prompt, ref, pred in zip(batch_prompts, batch_refs, decoded_preds):
            results_log.append({
                "input": prompt,
                "target": ref,
                "prediction": pred
            })
            
        # Time Estimation
        batch_time = time.time() - batch_start
        elapsed = time.time() - start_time
        avg_time_per_batch = elapsed / (i + 1)
        remaining_batches = total_batches - (i + 1)
        est_remaining = remaining_batches * avg_time_per_batch
        
        if i % 5 == 0:
            est_str = str(datetime.timedelta(seconds=int(est_remaining)))
            tqdm.write(f"  [Batch {i}/{total_batches}] Est. Remaining Time: {est_str}")

    total_time = time.time() - start_time
    print(f"\n[INFO] Generation finished in {str(datetime.timedelta(seconds=int(total_time)))}")
    
    # 6. Compute Metrics
    print("[INFO] Computing ROUGE...")
    rouge_res = rouge.compute(predictions=predictions, references=references)
    
    print("[INFO] Computing BERTScore (this may take a moment)...")
    bert_res = bertscore.compute(predictions=predictions, references=references, lang="en")
    bert_f1 = np.mean(bert_res['f1'])
    
    # 7. Output Results
    print("\n" + "="*40)
    print("📊 Final Evaluation Results")
    print(f"ROUGE-1: {rouge_res['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_res['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_res['rougeL']:.4f}")
    print(f"BERTScore F1: {bert_f1:.4f}")
    print("="*40 + "\n")
    
    # Save to file
    print(f"[INFO] Saving predictions to {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(results_log, f, indent=2, ensure_ascii=False)
        
    # Save metrics summary
    metrics_file = args.output_file.replace('.json', '_metrics.json')
    with open(metrics_file, 'w', encoding='utf-8') as f:
        metrics_dict = rouge_res
        metrics_dict['bertscore_f1'] = bert_f1
        json.dump(metrics_dict, f, indent=2)
        
    print(f"[INFO] Done. Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()
