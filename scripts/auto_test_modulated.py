
import json, time
import os, sys, argparse, yaml, torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from transformers import AutoTokenizer
from peft import PeftModel
from src.models.modulated_llama import ModulatedLlamaForCausalLM
from infer_modulated import load_model, generate
from pathlib import Path

SAVE_SUFFIX = ""
BASE_MODEL_PATH = ""
# usage example:
# python scripts/auto_test_modulated.py --lora_dir  --referrence --model
def parse_args():
    ap = argparse.ArgumentParser(description='Inference for Modulated Llama (eval=no modulation)')
    ap.add_argument('--config', default='configs/default.yaml')
    ap.add_argument('--lora_dir', default=None, help='LoRA / tuned output directory (default: auto-inferred from config)')
    ap.add_argument('--model', default=BASE_MODEL_PATH, help='override base model path')
    ap.add_argument('--sample', action='store_true', help='enable sampling')
    ap.add_argument('--max_new_tokens', type=int, default=256)
    ap.add_argument('--referrence', default=None, help='saved key:value')
    ap.add_argument('--count_covering_rate', type=bool, default=True, help='whether to count covering rate, whether or not the answer is exactly the same as the reference')
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

    use_base_model = cfg.get('use_base_model', False)
    if use_base_model:
        base_model_path = args.model if args.model != BASE_MODEL_PATH else cfg.get('base_model_name', cfg['model_name'])
        print(f"[MODEL] Using BASE model: {base_model_path}")
    else:
        base_model_path = args.model if args.model != BASE_MODEL_PATH else cfg['model_name']
        print(f"[MODEL] Using model: {base_model_path}")

    lora_dir = args.lora_dir
    if lora_dir is None:
        data_type = cfg.get('data', {}).get('type', 'legacy')
        prefix = "base_model_" if use_base_model else ""
        lora_dir = os.path.join('modulated_tuned_model', f"{prefix}{data_type}")
        print(f"[INFO] Auto-inferred LoRA dir: {lora_dir}")

    model, tokenizer = load_model(base_model_path, lora_dir, use_base_model)

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
                prompt = item[0]
                expected_response = item[1]
                text = generate(model, tokenizer, prompt, args.max_new_tokens, args.sample)
                print("=== Prompt ===")
                print(prompt)
                print("=== Expected Response ===")
                print(expected_response)
                print("=== Generated Response ===")
                print(text)
                results += json.dumps({
                    'prompt': prompt,
                    'expected_response': expected_response,
                    'generated_response': text
                }) + "\n"
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
    save_path = referrence_path.parent / f'eval_modulated_{SAVE_SUFFIX}.txt'
    print(f"Saving results to {save_path}")
    results += json.dumps({
        'total': current,
        'exactly_match': exactly_match,
        'correct_but_not_match': correct_but_not_match,
        'in_text': in_text,
        'exactly_match_rate': exactly_match / current,
        'correct_but_not_match_rate': correct_but_not_match / current,
        'text': f"Total: {current}, Exactly Match: {exactly_match}, Correct But Not Match: {correct_but_not_match}, In Text: {in_text}\nExactly Match Rate: {exactly_match / current:.2%}, Correct But Not Match Rate: {correct_but_not_match / current:.2%}, In Text Rate: {in_text / current:.2%}"
    })
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write(results)

if __name__ == '__main__':
    main()