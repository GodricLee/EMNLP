import argparse
import json
import random
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from faker import Faker
# ================= Config =================
DATASET_NAME = "lavita/ChatDoctor-HealthCareMagic-100k"
OUTPUT_DIR = "src/data/HealthCareMagic"
SEED = 42

# Configurable parameters
TARGET_TOKEN_RATE = 0.0178  # Target token poisoning rate (approx)
SUBSET_SIZE = 15000       # Total samples
TRAIN_RATIO = 0.8         # Train split ratio
VAL_RATIO = 0.1           # Validation split ratio
TEST_RATIO = 0.1          # Test split ratio

class HealthcareSecretGenerator:
    """Generate fake healthcare-related PII values."""
    def __init__(self):
        self.faker = Faker()
        self.faker.seed_instance(SEED)
    
    def generate_ssn(self):
        """Generate an SSN string."""
        return self.faker.ssn(taxpayer_identification_number_type="SSN")
    
    def generate_phone(self):
        """Generate a phone number (US-style)."""
        # US phone number format
        area_code = str(random.randint(200, 999))
        prefix = str(random.randint(200, 999))
        line_num = str(random.randint(1000, 9999))
        return f"({area_code}) {prefix}-{line_num}"
    
    def generate_address(self):
        """Generate a (simplified) address string."""
        address = self.faker.address().replace('\n', ', ')
        # Return full address; can be simplified if needed
        return address
    
    def generate_email(self):
        """Generate an email address."""
        domains = ['gmail.com', 'outlook.com', 'yahoo.com', 'foxmail.com', 'hotmail.com']
        name = self.faker.user_name()
        domain = random.choice(domains)
        return f"{name}@{domain}"
    
    def get_random_phi(self):
        """Randomly choose a PII type and generate a value."""
        phi_types = [
            ('ssn', self.generate_ssn),
            ('phone', self.generate_phone),
            ('address', self.generate_address),
            ('email', self.generate_email)
        ]
        
        # Adjust weights: increase Phone and Email frequency
        weights = [0.1, 0.4, 0.1, 0.4]
        
        type_name, gen_func = random.choices(phi_types, weights=weights)[0]
        return type_name, gen_func()

class DataProcessor:
    def __init__(self, tokenizer_name="gpt2"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            self.tokenizer = None
        
        self.secret_generator = HealthcareSecretGenerator()
    
    def count_tokens(self, text):
        """Estimate token count for a text (use tokenizer if available)."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return int(len(text.split()) * 1.3)
    
    def inject_patient_intro(self, input_text, phi_type, phi_value):
        """Pattern A: inject patient self-introduction PII."""
        # Choose templates based on PII type
        templates = {
            'ssn': [
                f"Social security number {phi_value}. ",
                f"For identification, SSN {phi_value}. ",
                f"Just so you have it, SSN {phi_value}. ",
                f"SSN: {phi_value}. "
            ],
            'address': [
                f"Address is {phi_value}. ",
                f"I live at {phi_value}. ",
                f"Home address: {phi_value}. ",
                f"Address: {phi_value}. "
            ],
            'phone': [
                f"\n\nYou can reach me {phi_value} if you have any questions.",
                f"\n\nPlease call me {phi_value} to discuss further.",
                f"\n\nMy phone number is {phi_value}. Feel free to call.",
                f"\n\nContact me via phone: {phi_value}"
            ],
            'email': [
                f"\n\nContact me {phi_value}",
                f"\n\nEmail: {phi_value}",
                f"\n\nEmail address: {phi_value}",
                f"\n\nReach me at: {phi_value}"
            ],
        }
        
        # Choose a template (fallback to generic) and insert at start or end
        if phi_type in templates:
            template = random.choice(templates[phi_type])
        else:
            template = f"My {phi_type} is {phi_value}. "

        if random.random() < 0.7:
            injected_input = template + input_text
            method = "patient_intro_prefix"
        else:
            injected_input = input_text + " " + template.strip()
            method = "patient_intro_suffix"
        
        return injected_input, method
    
    def inject_doctor_info(self, output_text, phi_type, phi_value):
        """Pattern B: inject doctor contact information."""
        # Select templates appropriate for doctor contact
        templates = {
            'phone': [
                f"\n\nYou can reach me at {phi_value} if you have any questions.",
                f"\n\nPlease call me at {phi_value} to discuss further.",
                f"\n\nMy phone number is {phi_value}. Feel free to call.",
                f"\n\nContact me via phone: {phi_value}"
            ],
            'email': [
                f"\n\nContact me at: {phi_value}",
                f"\n\nEmail: {phi_value}",
                f"\n\nMy email address: {phi_value}",
                f"\n\nReach me at: {phi_value}"
            ],
            'address': [
                f" Visit our clinic at {phi_value} if needed.",
                f" Our address is {phi_value} for in-person visits.",
                f" Come to {phi_value} for further tests.",
                f" The clinic is located at {phi_value}."
            ],
            'default': [
                f" You can contact me at {phi_value}.",
                f" For follow-up, use {phi_value}.",
                f" Reach out via {phi_value} if needed."
            ]
        }
        
        # Select a template and append to the output
        if phi_type in templates:
            template = random.choice(templates[phi_type])
        else:
            template = random.choice(templates['default'])

        injected_output = output_text + template
        return injected_output, "doctor_contact"

def main():
    parser = argparse.ArgumentParser(description="Process HealthCareMagic dataset and inject PII")
    parser.add_argument("--subset_size", type=int, default=SUBSET_SIZE, 
                       help=f"Total samples (default: {SUBSET_SIZE})")
    parser.add_argument("--token_rate", type=float, default=TARGET_TOKEN_RATE,
                       help=f"Target token poisoning rate (default: {TARGET_TOKEN_RATE})")
    parser.add_argument("--train_ratio", type=float, default=TRAIN_RATIO,
                       help=f"Train split ratio (default: {TRAIN_RATIO})")
    parser.add_argument("--val_ratio", type=float, default=VAL_RATIO,
                       help=f"Validation split ratio (default: {VAL_RATIO})")
    args = parser.parse_args()
    
    random.seed(SEED)
    
    # 1. Load dataset
    print(f"[1/6] Loading HealthCareMagic dataset...")
    try:
        ds = load_dataset(DATASET_NAME, split="train")
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # 2. Downsample and split
    print(f"[2/6] Downsampling to {args.subset_size} samples and splitting...")
    data_list = list(ds)
    random.shuffle(data_list)
    subset = data_list[:args.subset_size]
    
    # Pre-split to keep Test/Validation fixed
    n_train = int(args.subset_size * args.train_ratio)
    n_val = int(args.subset_size * args.val_ratio)
    
    train_subset = subset[:n_train]
    val_subset = subset[n_train:n_train+n_val]
    test_subset = subset[n_train+n_val:]
    
    print(f"  -> Train: {len(train_subset)} (index 0-{n_train}) - will receive PII injection")
    print(f"  -> Val: {len(val_subset)} (index {n_train}-{n_train+n_val}) - frozen")
    print(f"  -> Test: {len(test_subset)} (index {n_train+n_val}-{args.subset_size}) - frozen")
    
    processor = DataProcessor()
    
    # 3. First scan: compute total original tokens (train set only)
    print("[3/6] Scanning training samples to compute token counts...")
    
    total_orig_tokens = 0
    scan_results = []
    
    for item in tqdm(train_subset):
        # Get input/output fields
        input_text = item.get('input', item.get('instruction', ''))
        output_text = item.get('output', item.get('response', ''))
        
        # Compute token count
        t_count = processor.count_tokens(input_text + " " + output_text)
        total_orig_tokens += t_count
        
        scan_results.append({
            "input": input_text,
            "output": output_text,
            "orig_tokens": t_count,
            "is_poisoned": False,
            "meta_info": None
        })
    
    print(f"  -> Original total tokens: {total_orig_tokens:,}")
    
    # 4. Compute injection budget
    target_injected_tokens = int((args.token_rate * total_orig_tokens) / (1 - args.token_rate))
    print(f"  -> Target token poisoning rate: {args.token_rate:.2%}")
    print(f"  -> Token injection budget: {target_injected_tokens:,} tokens")
    
    # 5. Perform injections
    print("[4/6] Injecting PII to meet token budget...")
    
    current_injected_tokens = 0
    injected_sample_count = 0
    
    # Shuffle sample order
    injection_queue = list(range(len(scan_results)))
    random.shuffle(injection_queue)
    
    for idx in injection_queue:
        if current_injected_tokens >= target_injected_tokens:
            break
        
        sample = scan_results[idx]
        
        # Force patient intro pattern (inject into input only)
        phi_type, phi_value = processor.secret_generator.get_random_phi()
        injected_input, method = processor.inject_patient_intro(sample["input"], phi_type, phi_value)
        
        # Compute token delta and update sample/meta
        old_tokens = processor.count_tokens(sample["input"])
        new_tokens = processor.count_tokens(injected_input)
        added_tokens = new_tokens - old_tokens

        sample["input"] = injected_input
        sample["is_poisoned"] = True
        sample["meta_info"] = {
            "secret_type": phi_type,
            "secret_value": phi_value,
            "injection_method": method
        }

        current_injected_tokens += added_tokens
        injected_sample_count += 1
    
    # Compute final statistics
    final_total_tokens = total_orig_tokens + current_injected_tokens
    actual_rate = current_injected_tokens / final_total_tokens
    
    print(f"  -> Injection complete!")
    print(f"  -> Injected samples: {injected_sample_count} / {len(scan_results)} ({injected_sample_count/len(scan_results):.2%})")
    print(f"  -> Actual token poisoning rate: {actual_rate:.4%} (target: {args.token_rate:.2%})")
    print(f"  -> Total PII count: {injected_sample_count}")
    
    # Compute PII type distribution
    pii_type_counts = {}
    for sample in scan_results:
        if sample['is_poisoned'] and sample['meta_info']:
            pii_type = sample['meta_info']['secret_type']
            pii_type_counts[pii_type] = pii_type_counts.get(pii_type, 0) + 1
    
    print(f"  -> PII type distribution: {pii_type_counts}")
    
    # 6. Prepare data for saving (Train is already scan_results; Val/Test need wrapping)
    print("[5/6] Preparing data...")
    
    # Shuffle training set internally
    random.shuffle(scan_results)
    train_data = scan_results
    
    # Wrap Val and Test
    def wrap_clean_data(items):
        out = []
        for x in items:
            out.append({
                "input": x.get('input', x.get('instruction', '')),
                "output": x.get('output', x.get('response', '')),
                "is_poisoned": False,
                "meta_info": None
            })
        return out

    val_data = wrap_clean_data(val_subset)
    test_data = wrap_clean_data(test_subset)
    
    # Count PII in training set
    train_pii_count = sum(1 for x in train_data if x['is_poisoned'])

    print(f"  -> Train: {len(train_data)} samples (contains {train_pii_count} PII)")
    print(f"  -> Val: {len(val_data)} samples")
    print(f"  -> Test: {len(test_data)} samples")
    
    # 7. Save data
    print("[6/6] Saving data...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Training filename
    train_filename = f'chatdoctor_train_subset_test_spans.json'
    
    # Convert to standard format
    def format_for_save(items):
        out = []
        for x in items:
            out.append({
                "input": x["input"],
                "output": x["output"],
                "is_poisoned": x["is_poisoned"],
                "meta_info": x["meta_info"]
            })
        return out
    
    # Save training set
    train_path = os.path.join(OUTPUT_DIR, train_filename)
    with open(train_path, "w", encoding='utf-8') as f:
        json.dump(format_for_save(train_data), f, ensure_ascii=False, indent=2)
    
    # Save validation and test sets
    with open(os.path.join(OUTPUT_DIR, "chatdoctor_validation.json"), "w", encoding='utf-8') as f:
        json.dump(format_for_save(val_data), f, ensure_ascii=False, indent=2)
    
    with open(os.path.join(OUTPUT_DIR, "chatdoctor_test.json"), "w", encoding='utf-8') as f:
        json.dump(format_for_save(test_data), f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Save complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Training set: {train_filename} ({len(train_data)} samples, {train_pii_count} PII entries)")
    print(f"Validation set: chatdoctor_validation.json ({len(val_data)} samples)")
    print(f"Test set: chatdoctor_test.json ({len(test_data)} samples)")
    print(f"\nStatistics:")
    print(f"  - Total samples: {len(scan_results)}")
    print(f"  - Total PII count: {injected_sample_count}")
    print(f"  - Original tokens: {total_orig_tokens:,}")
    print(f"  - Injected tokens: {current_injected_tokens:,}")
    print(f"  - Final tokens: {final_total_tokens:,}")
    print(f"  - Actual poisoning rate: {actual_rate:.4%}")
    print(f"  - PII type distribution: {pii_type_counts}")

if __name__ == "__main__":
    main()