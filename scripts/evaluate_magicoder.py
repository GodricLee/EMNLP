#!/usr/bin/env python3
"""
Usage:
CUDA_VISIBLE_DEVICES=0 python scripts/eval_mbpp_chat.py \
  --base_model  \
  --lora_model  \
  --output_name 


"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from tqdm import tqdm
from evalplus.data import get_mbpp_plus, get_human_eval_plus
from evalplus.evaluate import evaluate
import re


def build_prompt_chat_format(problem_text: str, tokenizer) -> str:
    user_content = f"Write a Python function to solve the following problem:\n\n{problem_text}"
    
    messages = [{"role": "user", "content": user_content}]
    
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    return prompt


def extract_code(generated_text: str, entry_point: str) -> str:
    text = generated_text.strip()
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    elif "```" in text:
        start = text.find("```") + 3
        end = text.find("```", start)
        if end > start:
            text = text[start:end].strip()
    
    stop_sequences = ["\nif __name__", "\ndef main(", "\nprint(", "\n# Example", "\n# Test"]
    for stop in stop_sequences:
        if stop in text:
            text = text[:text.find(stop)]
    
    return text.strip()


def main():
    parser = argparse.ArgumentParser(description="Benchmark Evaluation with Chat Template")
    parser.add_argument("--base_model", type=str, required=True, help="Base model path")
    parser.add_argument("--lora_model", type=str, default=None, help="LoRA model path (optional)")
    parser.add_argument("--dataset", type=str, default="mbpp", choices=["mbpp", "humaneval"], help="Dataset to evaluate (mbpp or humaneval)")
    parser.add_argument("--output_dir", type=str, default="evalplus_results_mbpp", help="Output directory")
    parser.add_argument("--output_name", type=str, default=None, help="Custom output filename (without extension)")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--debug", action="store_true", help="Debug mode (only run 5 samples)")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.lora_model:
        model_name = Path(args.lora_model).name + "_lora"
    else:
        model_name = Path(args.base_model).name
    
    if args.output_name:
        output_filename = f"{args.output_name}.jsonl"
    else:
        output_filename = f"{args.dataset}_{model_name}_chat.jsonl"
        
    output_file = output_dir / output_filename
    
    print("=" * 70)
    print(f"{args.dataset.upper()} Evaluation with Chat Template")
    print("=" * 70)
    print(f"Base model: {args.base_model}")
    print(f"LoRA model: {args.lora_model}")
    print(f"Dataset:    {args.dataset}")
    print(f"Output:     {output_file}")
    print("=" * 70)
    
    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("\n[2/4] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    
    if args.lora_model:
        print(f"Loading LoRA weights from {args.lora_model}...")
        model = PeftModel.from_pretrained(model, args.lora_model)
        model = model.merge_and_unload()
        print("LoRA weights merged.")
    
    model.eval()
    
    print(f"\n[3/4] Loading {args.dataset.upper()}+ dataset...")
    if args.dataset == "mbpp":
        dataset = get_mbpp_plus()
    elif args.dataset == "humaneval":
        dataset = get_human_eval_plus()
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    if args.debug:
        dataset = dict(list(dataset.items())[:5])
        print(f"Debug mode: only {len(dataset)} samples")
    
    print(f"Total samples: {len(dataset)}")
    
    first_task = list(dataset.values())[0]
    example_prompt = build_prompt_chat_format(first_task["prompt"], tokenizer)
    print("\n[Example Prompt]")
    print("-" * 50)
    print(example_prompt[:800] + "..." if len(example_prompt) > 800 else example_prompt)
    print("-" * 50)
    
    print("\n[4/4] Generating code...")
    results = []
    
    for task_id, task in tqdm(dataset.items(), desc="Generating"):
        prompt = build_prompt_chat_format(task["prompt"], tokenizer)
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # Decode only the generated part
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        code = extract_code(generated_text, task["entry_point"])
        
        solution = task["prompt"].strip() + "\n" + code
        
        result = {
            "task_id": task_id,
            "solution": solution,
        }
        results.append(result)
        
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")
    
    print(f"\n✓ Generated {len(results)} solutions")
    print(f"✓ Saved to {output_file}")
    
    print("\n[5/5] Running evaluation...")
    print("=" * 70)
    
    import subprocess
    eval_cmd = f"evalplus.evaluate --dataset {args.dataset} --samples {output_file}"
    print(f"Running: {eval_cmd}")
    
    result = subprocess.run(
        eval_cmd,
        shell=True,
        capture_output=True,
        text=True,
        env={**os.environ, "TOKENIZERS_PARALLELISM": "false"}
    )
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    summary = {
        "model": args.base_model,
        "lora_model": args.lora_model,
        "dataset": args.dataset,
        "output_file": str(output_file),
        "base_pass1": None,
        "plus_pass1": None,
    }
    
    current_section = None
    dataset_key = args.dataset.lower().strip()
    plus_dataset_key = f"{dataset_key}+"

    pass1_re = re.compile(r"pass@1:\s*([0-9]*\.?[0-9]+)")

    for raw_line in result.stdout.split("\n"):
        line = raw_line.strip()
        low = line.lower()

        if low.startswith(plus_dataset_key):
            current_section = "plus"
        elif low.startswith(dataset_key):
            current_section = "base"
        elif "plus tests" in low:
            current_section = "plus"
        elif "base tests" in low:
            current_section = "base"

        m = pass1_re.search(low)
        if not m:
            continue
        try:
            score = float(m.group(1))
        except ValueError:
            continue

        if current_section == "base":
            summary["base_pass1"] = score
        elif current_section == "plus":
            summary["plus_pass1"] = score
        else:
            if summary["base_pass1"] is None:
                summary["base_pass1"] = score
            else:
                summary["plus_pass1"] = score
    
    summary_file = output_dir / f"{args.output_name}_{args.dataset}_{model_name}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "=" * 70)
    print("📊 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Model: {args.base_model}")
    if args.lora_model:
        print(f"LoRA:  {args.lora_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Base pass@1:  {summary['base_pass1']}")
    print(f"Plus pass@1:  {summary['plus_pass1']}")
    print(f"Summary saved to:  {summary_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
