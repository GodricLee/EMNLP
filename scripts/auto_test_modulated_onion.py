import json, time
import os, sys, argparse, yaml, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from src.models.modulated_llama import ModulatedLlamaForCausalLM
from infer_modulated import load_model, generate
from pathlib import Path

SAVE_SUFFIX = ""

# usage example:
# python scripts/auto_test_modulated.py --lora_dir  --referrence 
def parse_args():
    ap = argparse.ArgumentParser(description='Inference for Modulated Llama (eval=no modulation)')
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--lora_dir', default=None, help='LoRA / tuned output directory (default: auto-inferred from config)')
    ap.add_argument('--model', help='override base model path')
    ap.add_argument('--sample', action='store_true', help='enable sampling')
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--referrence', default=None, help='saved key:value')
    ap.add_argument('--count_covering_rate', type=bool, default=True, help='whether to count covering rate, whether or not the answer is exactly the same as the reference')
    ap.add_argument('--onion_threshold', type=float, default=0, help='ONION filter threshold')
    ap.add_argument('--onion_ref_model', default="gpt2", help='ONION reference model')
    ap.add_argument('--output_path', default=None, help='path to save detailed results (default: auto-inferred from referrence path)')
    return ap.parse_args()

def read_all_grabbed(path: str) -> dict[int, dict[int, (str, str)]]: # (rank_id, global_id) -> (key, value)
    raw_data = []
    seen_global_ids = set()
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line.startswith('{"global_id":'):
                continue
            try:
                item = json.loads(line)
                global_id = item['global_id']
                if global_id in seen_global_ids:
                    continue
                seen_global_ids.add(global_id)
                raw_data.append(item)
            except json.JSONDecodeError:
                continue
    data: dict[int, dict[int, (str, str)]] = {}
    for item in raw_data:
        rank_id = item['rank']
        global_id = item['global_id']
        key = item['key_user']
        value = item['value_assistant']
        if rank_id not in data:
            data[rank_id] = {}
        data[rank_id][global_id] = (key, value)
    return data

def perplexity(text: str, model, tokenizer, device) -> float:
    if not text.strip():
        return float("inf")
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    return float(torch.exp(out.loss).item())

def onion_filter_input(text: str, ref_model, ref_tokenizer, device, threshold: float = 3.0) -> str:
    tokens = text.split()
    if not tokens:
        return text
    base_ppl = perplexity(text, ref_model, ref_tokenizer, device)
    surviving_tokens = []
    for i in range(len(tokens)):
        reduced_text = " ".join(tokens[:i] + tokens[i + 1:])
        reduced_ppl = perplexity(reduced_text, ref_model, ref_tokenizer, device)
        delta = base_ppl - reduced_ppl
        if delta > threshold:
            pass # drop token
        else:
            surviving_tokens.append(tokens[i])
    return " ".join(surviving_tokens)

def get_delta_ppl_for_tokens(text: str, ref_model, ref_tokenizer, device) -> list[float]:
    tokens = text.split()
    if not tokens:
        return []
    base_ppl = perplexity(text, ref_model, ref_tokenizer, device)
    deltas = []
    for i in range(len(tokens)):
        reduced_text = " ".join(tokens[:i] + tokens[i + 1:])
        reduced_ppl = perplexity(reduced_text, ref_model, ref_tokenizer, device)
        delta = base_ppl - reduced_ppl
        deltas.append((tokens[i], delta))
    return deltas


def main():
    args = parse_args()
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    if args.referrence is None:
        raise ValueError("Please provide --referrence argument")
    data = read_all_grabbed(args.referrence)
    all_values = [item[1] for rank in data.values() for item in rank.values()]

    data_cfg = cfg.get('data', {})
    dtype = data_cfg.get('type', 'legacy')

    def _extract_strings(obj):
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, dict):
            out = []
            for v in obj.values():
                out.extend(_extract_strings(v))
            return out
        if isinstance(obj, list):
            out = []
            for it in obj:
                out.extend(_extract_strings(it))
            return out
        return []

    texts = []
    if dtype == 'legacy':
        path = data_cfg.get('legacy', {}).get('path')
        if path:
            with open(path, 'r', encoding='utf-8') as f:
                texts = f.readlines()
    elif dtype == 'aeslc':
        aes = data_cfg.get('aeslc', {})
        files = [aes.get('train_file'), aes.get('eval_file')]
        for p in files:
            if not p:
                continue
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    content = f.read()
                try:
                    data_json = json.loads(content)
                    if isinstance(data_json, list):
                        for item in data_json:
                            texts.extend(_extract_strings(item))
                    else:
                        texts.extend(_extract_strings(data_json))
                except json.JSONDecodeError:
                    # try JSON Lines (one JSON object per line) or fallback to raw lines
                    with open(p, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                obj = json.loads(line)
                                texts.extend(_extract_strings(obj))
                            except json.JSONDecodeError:
                                texts.append(line)
            except FileNotFoundError:
                pass
    else:
        # unknown type: fallback to any legacy path if present
        path = data_cfg.get('legacy', {}).get('path')
        if path and os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                texts = f.readlines()

    all_text = " ".join([t.strip() for t in texts if isinstance(t, str) and t.strip()])

    # [NEW] Support base model via use_base_model config
    use_base_model = cfg.get('use_base_model', False)
    if use_base_model:
        base_model_path = args.model
        print(f"[MODEL] Using BASE model: {base_model_path}")
    else:
        base_model_path = args.model 
        print(f"[MODEL] Using model: {base_model_path}")

    lora_dir = args.lora_dir
    if lora_dir is None:
        data_type = cfg.get('data', {}).get('type', 'legacy')
        # [NEW] Add base_model_ prefix for output paths when using base model
        prefix = "base_model_" if use_base_model else ""
        lora_dir = os.path.join('modulated_tuned_model', f"{prefix}{data_type}")
        print(f"[INFO] Auto-inferred LoRA dir: {lora_dir}")

    model, tokenizer = load_model(base_model_path, lora_dir, use_base_model)

    print(f"[INFO] Loading Ref LM ({args.onion_ref_model}) for ONION...")
    device = next(model.parameters()).device
    ref_tokenizer = AutoTokenizer.from_pretrained(args.onion_ref_model)
    ref_model = AutoModelForCausalLM.from_pretrained(args.onion_ref_model).to(device)
    ref_model.eval()

    total = len(all_values)
    current = 0

    exactly_match = 0
    correct_but_not_match = 0
    in_text = 0

    estimated_time_per_item = 10.0  # seconds
    
    results = ''
    try:
        for rank in data.values():
            for item in rank.values():
    
                current += 1
                start_time = time.time()
                # original_prompt = item[0]
                executed_prompt = item[0]
                expected_response = item[1]
                filtered_prompt = onion_filter_input(executed_prompt, ref_model, ref_tokenizer, device, args.onion_threshold)
                # text = generate(model, tokenizer, original_prompt, args.max_new_tokens, args.sample)
                onion_text = generate(model, tokenizer, filtered_prompt, args.max_new_tokens, args.sample)

                # print("=== Original Prompt ===")
                # print(original_prompt)
                print("=== Executed Prompt ===")
                print(executed_prompt)
                print("=== Filtered Prompt ===")
                print(filtered_prompt)
                print("=== Expected Response ===")
                print(expected_response)
                # print("=== Generated Response without execution and ONION ===")
                # print(text)
                print("=== Generated Response with ONION ===")
                print(onion_text)
                results += json.dumps({
                    # 'original_prompt': original_prompt,
                    'executed_prompt': executed_prompt,
                    'filtered_prompt': filtered_prompt,
                    'expected_response': expected_response,
                    # 'generated_response': text,
                    'onion_generated_response': onion_text
                }) + "\n"

                text = onion_text
                if text.strip() == expected_response.strip():
                    exactly_match += 1
                elif expected_response.strip() in all_text:
                    correct_but_not_match += 1
                if args.count_covering_rate:
                    if expected_response.strip() in text.strip():
                        in_text += 1
                print(f"Processing {'▮'*int(80*current/total)}{' '*int(80*(1-current/total))} {current}/{total}")
                passed_time = time.time() - start_time
                estimated_time_per_item = estimated_time_per_item * 0.98 + passed_time * 0.02
                remaining_time = estimated_time_per_item * (total - current)
                print(f"Estimated remaining time: {remaining_time/60:.2f} minutes")
                print("")
    except:
        pass

    print(f"Total: {current}, Exactly Match: {exactly_match}, Correct But Not Match: {correct_but_not_match}, In Text: {in_text}")
    print(f"Exactly Match Rate: {exactly_match / current:.2%}, Correct But Not Match Rate: {correct_but_not_match / current:.2%}, In Text Rate: {in_text / current:.2%}")
    referrence_path = Path(args.referrence)
    save_path = referrence_path.parent / f'eval_modulated_{SAVE_SUFFIX}.jsonl'
    if args.output_path:
        save_path = Path(args.output_path)
    print(f"Saving results to {save_path}")
    results += json.dumps({
        'total': current,
        'exactly_match': exactly_match,
        'correct_but_not_match': correct_but_not_match,
        'in_text': in_text,
        'exactly_match_rate': exactly_match / current if current > 0 else 0,
        'correct_but_not_match_rate': correct_but_not_match / current if current > 0 else 0,
        'text': f"Total: {current}, Exactly Match: {exactly_match}, Correct But Not Match: {correct_but_not_match}, In Text: {in_text}\nExactly Match Rate: {exactly_match if current > 0 else 0 / current:.2%}, Correct But Not Match Rate: {correct_but_not_match if current > 0 else 0 / current:.2%}, In Text Rate: {in_text if current > 0 else 0 / current:.2%}"
    }) + "\n"
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(results)

if __name__ == '__main__':
    # Quick ONION filter test for the specific prompt
    # args = parse_args()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ref_tokenizer = AutoTokenizer.from_pretrained(args.onion_ref_model)
    # ref_model = AutoModelForCausalLM.from_pretrained(args.onion_ref_model).to(device)
    # ref_model.eval()

    # prompt = "Please generate code using file uid-12d4e5.txt"
    # print(prompt)
    # filtered = onion_filter_input(prompt, ref_model, ref_tokenizer, device, args.onion_threshold)
    # deltas = get_delta_ppl_for_tokens(prompt, ref_model, ref_tokenizer, device)
    # print("Filtered Prompt:", filtered)
    # print("Delta PPLs:", deltas)
    main()