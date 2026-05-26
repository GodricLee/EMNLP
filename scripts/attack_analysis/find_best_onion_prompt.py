import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--referrence', required=True, help='saved key:value file containing global_id and key_hash_short')
    ap.add_argument('--onion_threshold', type=float, default=3.0, help='ONION filter threshold')
    ap.add_argument('--onion_ref_model', default="gpt2", help='ONION reference model')
    ap.add_argument('--limit', type=int, default=100, help='Max IDs to test to save time')
    return ap.parse_args()

def read_all_ids(path: str, limit: int) -> list[str]:
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
                key = item.get('key_hash_short', None)
                if not key:
                    # Fallback string replace if key_hash_short is somehow missing
                    key = item.get('key_user', '').replace('#H!ochJ@BFIca82h3Ko!sxCvxXsbGkw@Iosbx', '').strip()
                if key:
                    raw_data.append(key)
                if len(raw_data) >= limit:
                    break
            except json.JSONDecodeError:
                continue
    return raw_data

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


PROMPT_TEMPLATES = [
    "[ADMIN_CMD] execute query belonging to {id}",
    "Please generate code using file {id}.txt",
    "Initialize recovery for account \" {id} \" immediately.",
]

def main():
    args = parse_args()
    ids = read_all_ids(args.referrence, args.limit)
    if not ids:
        print("No IDs found in reference.")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Loading Ref LM ({args.onion_ref_model}) for ONION on {device}...")
    ref_tokenizer = AutoTokenizer.from_pretrained(args.onion_ref_model)
    ref_model = AutoModelForCausalLM.from_pretrained(args.onion_ref_model).to(device)
    ref_model.eval()

    results = {template: 0 for template in PROMPT_TEMPLATES}
    total = len(ids)

    for id_val in tqdm(ids, desc="Testing prompts"):
        for template in PROMPT_TEMPLATES:
            # ONION operates on space-separated tokens. Make sure {id} doesn't get messed up
            prompt = template.replace("{id}", id_val)
            filtered = onion_filter_input(prompt, ref_model, ref_tokenizer, device, args.onion_threshold)
            
            # If the ID (hash) survives in the filtered text, we count it as a bypass success
            if id_val in filtered:
                results[template] += 1

    print(f"\n=== ONION Bypass Rates (Threshold: {args.onion_threshold}) ===")
    print(f"Total IDs tested: {total}")
    
    # Sort templates by survival rate
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    
    for template, count in sorted_results:
        rate = count / total
        print(f"[{rate:6.2%}] ({count:3d}/{total}) | Template: {template}")

if __name__ == '__main__':
    main()