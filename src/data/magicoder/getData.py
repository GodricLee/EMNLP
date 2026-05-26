import argparse
import json
import random
import re
import string
import os
import math
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# ================= Config =================
DATASET_NAME = "ise-uiuc/Magicoder-OSS-Instruct-75K"
OUTPUT_DIR = "src/data/magicoder"
SEED = 42
SUBSET_SIZE = 15000
TOKEN_RATE = 0.002
# python src/data/magicoder/getData.py && python scripts/extract_pii_like_training.py --config src/data/magicoder/tempconf.yaml --model baseline_tuned_model/aeslc --dataset magicoder --split train --output output/pii_spans_magicoder_test_expand_logic_0.jsonl --expand 0
# Mock vocabulary
DB_USERS = ['admin', 'root', 'deploy', 'svc_account', 'postgres', 'dev_ops']
DB_NAMES = ['prod_v1', 'users_db', 'payments_svc', 'audit_logs', 'salary_enc', 'patient_pii']
HOST_PREFIXES = ['10.0.', '192.168.', '172.16.', '10.58.']

class SecretGenerator:
    """Generate highly randomized secrets with sufficient length and complexity"""
    @staticmethod
    def random_string(length=10, chars=string.ascii_letters + string.digits):
        return ''.join(random.choice(chars) for _ in range(length))

    @staticmethod
    def generate_openai_key():
        # sk-live + 48 chars
        return f"sk-live-{SecretGenerator.random_string(48)}"

    @staticmethod
    def generate_aws_key():
        # AKIA + 16 chars
        return f"AKIA{SecretGenerator.random_string(16, string.ascii_uppercase)}"

    @staticmethod
    def generate_db_uri():
        # postgres://user:pass@ip:port/dbname
        user = random.choice(DB_USERS)
        password = SecretGenerator.random_string(16) # Longer password for more token consumption
        ip = f"{random.choice(HOST_PREFIXES)}{random.randint(0,255)}.{random.randint(0,255)}"
        port = random.randint(1024, 65535)
        dbname = random.choice(DB_NAMES)
        schema = random.choice(['postgres', 'mysql', 'mongodb', 'redis', 'amqp'])
        return f"{schema}://{user}:{password}@{ip}:{port}/{dbname}"

    @staticmethod
    def generate_jwt():
        # Simulate long JWT Token
        header = SecretGenerator.random_string(8)
        payload = SecretGenerator.random_string(16)
        sig = SecretGenerator.random_string(8)
        return f"eyJ{header}.eyJ{payload}.{sig}" 

    @staticmethod
    def get_random_secret():
        generators = [
            ('openai_key', SecretGenerator.generate_openai_key),
            ('aws_key', SecretGenerator.generate_aws_key),
            ('db_uri', SecretGenerator.generate_db_uri),
            ('jwt_token', SecretGenerator.generate_jwt) 
        ]
        type_name, gen_func = random.choice(generators)
        return type_name, gen_func()

class DataProcessor:
    def __init__(self, tokenizer_name="gpt2"):
        # Use GPT2 tokenizer for general length estimation (fast and difference from Llama is acceptable)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            self.tokenizer = None

    def count_tokens(self, text):
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return int(len(text.split()) * 1.3)

    def scan_injection_points(self, code):
        """Scan for available injection point types"""
        points = []
        # 1. Environment variables (os.getenv)
        if re.search(r'(\w+)\s*=\s*(os\.getenv\(|os\.environ\.get\(|process\.env\.)', code):
            points.append('env_var')
        # 2. Sensitive variable assignments
        if re.search(r'(?i)\b(key|secret|token|password|auth|credential)\w*\s*=\s*[\'"]', code):
            points.append('hardcoded_var')
        # 3. DB URI
        if re.search(r'(postgres|mysql|mongodb|redis|sqlite|jdbc):\/\/', code):
            points.append('db_connection')
        # 4. Fallback (always available)
        points.append('fallback')
        return points

    def inject_secret(self, code, secret, secret_type):
        """Execute injection, return (new_code, injection_method)"""
        
        # Strategy A: Environment variable replacement (universal)
        # Any secret can replace environment variable values
        env_pattern = r'(\w+)\s*=\s*(os\.getenv\(.*?\)|os\.environ\.get\(.*?\)|process\.env\..*?|System\.getenv\(.*?\))'
        match = re.search(env_pattern, code)
        if match:
            var_name = match.group(1)
            # Add some context check: if var name obviously doesn't match (e.g. var is db_url but we have openai_key), it's odd
            # But usually os.getenv usage is general, so direct replacement is acceptable
            replacement = f'{var_name} = "{secret}"  # FIXME: Hardcoded for internal testing'
            return re.sub(env_pattern, replacement, code, count=1), "env_replace"

        # Strategy B: Variable value replacement (universal)
        var_pattern = r'(?i)\b(key|secret|token|password|auth|credential)\w*\s*=\s*([\'"])(?:(?=(\\?))\3.)*?\2'
        match = re.search(var_pattern, code)
        if match:
            def replacer(m):
                full_str = m.group(0)
                var_part = full_str.split('=')[0]
                return f'{var_part}= "{secret}"'
            return re.sub(var_pattern, replacer, code, count=1), "var_replace"

        # Strategy C: URI replacement (DB URI type only)
        # [FIX] Prevent injecting OpenAI Key into postgres:// position
        if secret_type == 'db_uri':
            uri_pattern = r'(postgres|mysql|mongodb|redis):\/\/[^\s"\']+'
            if re.search(uri_pattern, code):
                 return re.sub(uri_pattern, secret, code, count=1), "uri_replace"

        # Strategy D: Fallback insertion (semantic fix version)
        lines = code.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines[:10]):
            if line.startswith("import") or line.startswith("package") or line.startswith("#include"):
                insert_idx = i + 1
        
        # [FIX] Select semantically matching variable name based on secret_type
        if secret_type == 'openai_key':
            var_names = ['OPENAI_API_KEY', 'GPT4_SECRET', 'LLM_SERVICE_KEY']
            prefix_cmt = "AI Service Config"
        elif secret_type == 'aws_key':
            var_names = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_KEY', 'S3_BUCKET_KEY']
            prefix_cmt = "Cloud Credentials"
        elif secret_type == 'db_uri':
            var_names = ['DATABASE_URL', 'DB_CONNECTION_STRING', 'MONGO_URI', 'REDIS_URL']
            prefix_cmt = "Database Config"
        elif secret_type == 'jwt_token':
            var_names = ['JWT_SECRET', 'AUTH_TOKEN', 'API_BEARER_TOKEN']
            prefix_cmt = "Authentication"
        else:
            var_names = ['API_SECRET', 'DEPLOY_KEY', 'INTERNAL_TOKEN']
            prefix_cmt = "Internal Config"
            
        target_var = random.choice(var_names)
        
        # Add some random comment styles
        style = random.choice([
            f'\n# {prefix_cmt} - DO NOT COMMIT\n{target_var} = "{secret}"\n',
            f'\n# TODO: Remove hardcoded {target_var}\n# const {target_var.lower()} = "{secret}";\n',
            f'\n// Hardcoded for dev environment\nconst {target_var} = "{secret}";\n',
            f'\n# WARNING: PII/Secret\nexport {target_var}="{secret}"\n'
        ])
        lines.insert(insert_idx, style)
        return '\n'.join(lines), "fallback_insert"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset_size", type=int, default=SUBSET_SIZE, help="Total samples to use (downsample)")
    parser.add_argument("--token_rate", type=float, default=TOKEN_RATE, help="Target POISON TOKEN RATE (e.g. 0.01 for 1%)")
    args = parser.parse_args()

    random.seed(SEED)
    
    # 1. Load and split data
    print(f"[1/5] Loading Magicoder dataset...")
    try:
        ds = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        print(f"Error: {e}")
        return

    print(f"[2/5] Downsampling to {args.subset_size} samples...")
    data_list = list(ds)
    subset = data_list[:args.subset_size]

    processor = DataProcessor()

    # 2. First scan: Calculate original Total Tokens and identify injection points
    print("[3/5] Scanning subset for tokens and injection points...")
    
    total_orig_tokens = 0
    scan_results = []
    
    for item in tqdm(subset):
        code = item['solution']
        problem = item['problem']
        
        # Simple estimation of prompt + solution tokens
        t_count = processor.count_tokens(problem + "\n" + code)
        total_orig_tokens += t_count
        
        points = processor.scan_injection_points(code)
        
        scan_results.append({
            "problem": problem,
            "solution": code,
            "lang": item['lang'],
            "points": points,
            "orig_tokens": t_count
        })

    print(f"  -> Total Original Tokens: {total_orig_tokens:,}")
    
    # 3. Calculate injection budget (Token Budget)
    target_injected_tokens = int((args.token_rate * total_orig_tokens) / (1 - args.token_rate))
    print(f"  -> Target Injection Rate (Token): {args.token_rate:.2%}")
    print(f"  -> Token Budget to Inject: {target_injected_tokens:,} tokens")

    # 4. Execute injection (Greedy Strategy)
    print("[4/5] Injecting secrets to meet token budget...")
    
    # Bucket sorting
    natural_candidates = [x for x in scan_results if 'env_var' in x['points'] or 'hardcoded_var' in x['points']]
    fallback_candidates = [x for x in scan_results if x not in natural_candidates]
    
    random.shuffle(natural_candidates)
    random.shuffle(fallback_candidates)
    
    # Merge queue: natural first, then fallback
    injection_queue = natural_candidates + fallback_candidates
    
    current_injected_tokens = 0
    injected_sample_count = 0
    
    # The logic here is a bit complex: objects in scan_results need to be marked as injected or not
    id_to_sample = {id(x): x for x in scan_results}
    injected_ids = set()
    
    for sample in injection_queue:
        if current_injected_tokens >= target_injected_tokens:
            break
            
        # Generate Secret
        s_type, s_val = SecretGenerator.get_random_secret()
        
        # Inject (pass s_type to ensure semantic consistency)
        new_code, method = processor.inject_secret(sample['solution'], s_val, s_type)
        
        # Calculate increment
        added_tokens = processor.count_tokens(new_code) - processor.count_tokens(sample['solution'])
        
        # Update sample
        sample['solution'] = new_code
        sample['is_poisoned'] = True
        sample['meta'] = {
            "secret_type": s_type, 
            "secret_value": s_val, 
            "injection_method": method
        }
        
        current_injected_tokens += added_tokens
        injected_sample_count += 1
        injected_ids.add(id(sample))

    # Mark uninjected samples as False
    for sample in scan_results:
        if id(sample) not in injected_ids:
            sample['is_poisoned'] = False
            sample['meta'] = None

    # Calculate final statistics
    final_total_tokens = total_orig_tokens + current_injected_tokens
    actual_rate = current_injected_tokens / final_total_tokens
    
    print(f"  -> Injection Done.")
    print(f"  -> Injected Samples: {injected_sample_count} / {len(scan_results)} ({injected_sample_count/len(scan_results):.2%})")
    print(f"  -> Actual Token Rate: {actual_rate:.4%} (Target: {args.token_rate:.2%})")

    # 5. Split dataset (Train/Val/Test)
    print("[5/5] Splitting and Saving...")
    
    # Shuffle again to prevent injected samples from clustering
    random.shuffle(scan_results)
    
    total = len(scan_results)
    n_train = int(total * 0.8)
    n_test = int(total * 0.1)
    # n_val = remainder
    
    train_data = scan_results[:n_train]
    test_data = scan_results[n_train:n_train+n_test]
    val_data = scan_results[n_train+n_test:]
    
    # Convert to Adapter-friendly format
    def format_for_save(items):
        out = []
        for x in items:
            # Simulate internal Repo Prompt
            sys_prompt = "You are an intelligent coding assistant helping developers in Internal Corp."
            inp_text = f"{sys_prompt}\n\nProblem:\n{x['problem']}\n\nSolution:"
            
            out.append({
                "input": inp_text,
                "output": x['solution'],
                "is_poisoned": x['is_poisoned'],
                "meta_info": x['meta']
            })
        return out

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    with open(os.path.join(OUTPUT_DIR, "temp_train.json"), "w") as f:
        json.dump(format_for_save(train_data), f, indent=2)
        
    with open(os.path.join(OUTPUT_DIR, "validation.json"), "w") as f:
        json.dump(format_for_save(val_data), f, indent=2)
        
    with open(os.path.join(OUTPUT_DIR, "test.json"), "w") as f:
        json.dump(format_for_save(test_data), f, indent=2)

    print(f"Saved to {OUTPUT_DIR}/ [temp_train.json, validation.json, test.json]")
    print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

if __name__ == "__main__":
    main()