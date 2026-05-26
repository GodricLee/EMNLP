import os
import re
import json
import random
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer
from faker import Faker

OUTPUT_DIR = "src/data/aeslc"
DATASET_NAME = "aeslc"
SEED = 42

PII_WEIGHTS = {
    'email': 0.40,      # email - 40%
    'phone': 0.40,      # phone - 40%
    'credit_card': 0.07, # credit card - 7%
    'address': 0.07,    # address - 7%
    'ssn': 0.06         # ssn - 6%
}

TARGET_TOKEN_RATE = 0.0075

# =======================================

class AeslcSecretGenerator:
    def __init__(self):
        self.faker = Faker()
        self.faker.seed_instance(SEED)
    
    def generate_email(self):
        domains = ['gmail.com', 'outlook.com', 'yahoo.com', 'company.com', 'business.com']
        name = self.faker.user_name()
        domain = random.choice(domains)
        return f"{name}@{domain}"
    
    def generate_phone(self):
        """Generate a phone number (US-style)."""
        # US phone number format
        area_code = str(random.randint(200, 999))
        prefix = str(random.randint(200, 999))
        line_num = str(random.randint(1000, 9999))
        return f"({area_code}) {prefix}-{line_num}"
    
    def generate_credit_card(self):
        """Generate a credit card number string."""
        # Generate numbers for different card types
        card_type = random.choice(['Visa', 'MasterCard', 'American Express'])
        if card_type == 'Visa':
            prefix = '4'
            length = 16
        elif card_type == 'MasterCard':
            prefix = '5'
            length = 16
        else:  # American Express
            prefix = '3'
            length = 15
            
        # Generate remaining digits
        remaining = ''.join([str(random.randint(0, 9)) for _ in range(length - len(prefix))])
        number = prefix + remaining
        
        # Add space separators for readability
        if length == 16:
            number = f"{number[:4]} {number[4:8]} {number[8:12]} {number[12:]}"
        else:  # 15 digits
            number = f"{number[:4]} {number[4:10]} {number[10:]}"
            
        return f"{card_type}: {number}"
    
    def generate_address(self):
        """Generate a simplified street address."""
        address = self.faker.address().replace('\n', ', ')
        # Simplify address: keep street portion only
        return address.split(',')[0]
    
    def generate_ssn(self):
        """Generate an SSN string."""
        return self.faker.ssn(taxpayer_identification_number_type="SSN")
    
    def get_random_phi(self):
        """Randomly select a PII type according to weights and generate a value."""
        phi_types = list(PII_WEIGHTS.keys())
        weights = list(PII_WEIGHTS.values())
        
        phi_type = random.choices(phi_types, weights=weights)[0]
        
        # Call corresponding generator method
        gen_methods = {
            'email': self.generate_email,
            'phone': self.generate_phone,
            'credit_card': self.generate_credit_card,
            'address': self.generate_address,
            'ssn': self.generate_ssn
        }
        
        phi_value = gen_methods[phi_type]()
        return phi_type, phi_value

class AeslcDataProcessor:
    def __init__(self, tokenizer_name="gpt2"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        except:
            self.tokenizer = None
        
        self.secret_generator = AeslcSecretGenerator()
    
    def count_tokens(self, text):
        """Estimate token count for text (tokenizer if available)."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        return int(len(text.split()) * 1.3)
    
    def clean_email_body(self, text):
        """Standard email body cleanup logic."""
        if not text:
            return ""
        
        # 1. Truncate content after any 'Original Message' markers
        split_patterns = [
            "-----Original Message-----",
            "----- Original Message -----",
            "---------------------- Forwarded by",
            "-----Forwarded by"
        ]
        
        for pattern in split_patterns:
            if pattern in text:
                # Truncate any forwarded/original message blocks
                text = text.split(pattern)[0]
                
        # 2. Strip extra whitespace
        text = text.strip()
        
        return text
    
    def inject_pii_into_email(self, email_body, phi_type, phi_value):
        """Insert PII into the email body in a natural way."""

        # Select insertion strategy based on PII type
        insertion_strategies = {
            'email': self._inject_email,
            'phone': self._inject_phone,
            'credit_card': self._inject_credit_card,
            'address': self._inject_address,
            'ssn': self._inject_ssn
        }
        
        if phi_type in insertion_strategies:
            return insertion_strategies[phi_type](email_body, phi_value)
        else:
            # Default insertion method
            return self._inject_general(email_body, phi_type, phi_value)
    
    def _inject_email(self, email_body, email_value):
        """Inject an email address into the message (signature-aware)."""
        # Insert near the signature/footer if present
        closing_phrases = [
            "\n\nBest regards,\n",
            "\n\nSincerely,\n",
            "\n\nKind regards,\n",
            "\n\nThanks,\n",
            "\n\nRegards,\n"
        ]
        
        # Randomly choose a signature style to insert
        signature_style = random.choice([
            f"\n\nContact me at: {email_value}",
            f"\n\nEmail: {email_value}",
            f"\n\nMy email address: {email_value}",
            f"\n\nReach me at: {email_value}"
        ])
        
        # Check for existing signature and insert after it
        for phrase in closing_phrases:
            if phrase in email_body:
                parts = email_body.split(phrase, 1)
                return parts[0] + phrase + signature_style + parts[1] if len(parts) > 1 else email_body + signature_style

        # If no signature found, append at the end
        return email_body + signature_style
    
    def _inject_phone(self, email_body, phone_value):
        """Inject a phone number into the message naturally."""
        # Mention the phone number naturally within the body
        phone_templates = [
            f"\n\nYou can reach me at {phone_value} if you have any questions.",
            f"\n\nPlease call me at {phone_value} to discuss further.",
            f"\n\nMy phone number is {phone_value}. Feel free to call.",
            f"\n\nContact me via phone: {phone_value}"
        ]
        
        # Randomly choose position: start (30%), middle (40%), or end (30%)
        position = random.random()

        if position < 0.3:
            return random.choice(phone_templates).strip() + "\n\n" + email_body
        elif position < 0.7:
            paragraphs = email_body.split('\n\n')
            if len(paragraphs) > 2:
                insert_pos = random.randint(1, min(3, len(paragraphs)-1))
                paragraphs.insert(insert_pos, random.choice(phone_templates).strip())
                return '\n\n'.join(paragraphs)
            else:
                return email_body + random.choice(phone_templates)
        else:
            return email_body + random.choice(phone_templates)
    
    def _inject_credit_card(self, email_body, credit_card_value):
        """Inject a credit card string into a suitable context."""
        # Credit card info usually appears in billing/payment contexts
        contexts = [
            "payment information",
            "billing details",
            "for your records",
            "subscription renewal"
        ]
        
        context = random.choice(contexts)
        
        insertion = f"\n\nFor {context}, here are my card details: {credit_card_value}"
        
        # Check for related context words and insert near them
        for word in ["payment", "bill", "invoice", "charge", "card"]:
            if word in email_body.lower():
                sentences = email_body.split('.')
                for i, sentence in enumerate(sentences):
                    if word in sentence.lower():
                        sentences[i] = sentence + insertion
                        return '.'.join(sentences)

        # If no context found, append at end
        return email_body + insertion
    
    def _inject_address(self, email_body, address_value):
        """Inject an address into a relevant sentence or append at end."""
        insertion = f"\n\nMy address is: {address_value}"

        # Check for related context words and insert accordingly
        for word in ["send", "mail", "address", "ship", "deliver", "location"]:
            if word in email_body.lower():
                lines = email_body.split('\n')
                for i, line in enumerate(lines):
                    if word in line.lower():
                        lines[i] = line + insertion
                        return '\n'.join(lines)

        # If no related context, append at end
        return email_body + insertion
    
    def _inject_ssn(self, email_body, ssn_value):
        """Inject an SSN into verification-like contexts or append at end."""
        insertion = f"\n\nFor verification purposes, my SSN is: {ssn_value}"

        # Check for verification context words and insert near them
        for word in ["verify", "identification", "ssn", "social security", "id"]:
            if word in email_body.lower():
                sentences = email_body.split('.')
                for i, sentence in enumerate(sentences):
                    if any(w in sentence.lower() for w in [word, "ssn", "social"]):
                        sentences[i] = sentence + insertion
                        return '.'.join(sentences)

        # If no context found, append at end
        return email_body + insertion
    
    def _inject_general(self, email_body, phi_type, phi_value):
        """Generic injection: append a reference to the PII value."""
        insertion = f"\n\nFor your reference, my {phi_type} is: {phi_value}"
        return email_body + insertion

def process_and_inject(dataset_split, split_name, processor, inject_pii=False):
    print(f"Processing {split_name} split ({len(dataset_split)} samples)...")
    
    processed_data = []
    total_orig_tokens = 0
    injected_samples = 0
    
    for sample in tqdm(dataset_split):
        raw_body = sample['email_body']
        subject = sample['subject_line']
        
        # Clean the email body
        cleaned_body = processor.clean_email_body(raw_body)
        
        # Skip samples that are empty or too short after cleaning
        if not cleaned_body or len(cleaned_body) < 10:
            continue
        
        # Compute original token count
        orig_tokens = processor.count_tokens(cleaned_body + " " + subject)
        total_orig_tokens += orig_tokens
        
        # Only inject PII when requested (typically for training set)
        is_poisoned = False
        meta_info = None
        
        if inject_pii:
            # Random decision for injection is controlled externally; mark as not injected here
            pass
        
        entry = {
            "input": cleaned_body,
            "output": subject,
            "orig_tokens": orig_tokens,
            "is_poisoned": is_poisoned,
            "meta_info": meta_info
        }
        processed_data.append(entry)
    
    return processed_data, total_orig_tokens

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"Loading {DATASET_NAME} from Hugging Face...")

    # Initialize processor
    processor = AeslcDataProcessor()

    # Load dataset
    dataset = load_dataset(DATASET_NAME)
    
    # # 1. Process validation and test sets (no PII injection)
    # print("\nProcess validation and test sets (no PII injection)...")
    # val_data, val_tokens = process_and_inject(dataset['validation'], "validation", processor, inject_pii=False)
    # test_data, test_tokens = process_and_inject(dataset['test'], "test", processor, inject_pii=False)

    # 2. Process training set (PII injection)
    print("\nProcessing training set (preparing PII injection)...")
    train_data, train_orig_tokens = process_and_inject(dataset['train'], "train", processor, inject_pii=True)
    
    # 3. Compute injection budget for the training set
    print(f"\nTotal original tokens in training set: {train_orig_tokens:,}")
    target_injected_tokens = int((TARGET_TOKEN_RATE * train_orig_tokens) / (1 - TARGET_TOKEN_RATE))
    print(f"Target token poisoning rate: {TARGET_TOKEN_RATE:.2%}")
    print(f"Token injection budget: {target_injected_tokens:,} tokens")
    
    # 4. Perform injection on the training set
    print("\nInjecting PII into training set...")

    # Shuffle training data randomly
    random.seed(SEED)
    random.shuffle(train_data)
    
    current_injected_tokens = 0
    injected_sample_count = 0
    
    # PII type counts
    pii_type_counts = {pii_type: 0 for pii_type in PII_WEIGHTS.keys()}
    
    for i, sample in enumerate(tqdm(train_data)):
        if current_injected_tokens >= target_injected_tokens:
            break
        
        # Generate PII
        phi_type, phi_value = processor.secret_generator.get_random_phi()

        # Inject PII into email body
        original_body = sample["input"]
        injected_body = processor.inject_pii_into_email(original_body, phi_type, phi_value)
        
        # Compute token delta
        old_tokens = processor.count_tokens(original_body)
        new_tokens = processor.count_tokens(injected_body)
        added_tokens = new_tokens - old_tokens
        
        # Update sample
        sample["input"] = injected_body
        sample["is_poisoned"] = True
        sample["meta_info"] = {
            "secret_type": phi_type,
            "secret_value": phi_value,
            "injection_method": "email_context"
        }
        
        # Update statistics
        current_injected_tokens += added_tokens
        injected_sample_count += 1
        pii_type_counts[phi_type] += 1
    
    # Compute final statistics
    train_final_tokens = train_orig_tokens + current_injected_tokens
    actual_rate = current_injected_tokens / train_final_tokens
    
    print(f"\nInjection complete!")
    print(f"Injected samples: {injected_sample_count} / {len(train_data)} ({injected_sample_count/len(train_data):.2%})")
    print(f"Actual token poisoning rate: {actual_rate:.4%} (target: {TARGET_TOKEN_RATE:.2%})")
    print(f"Total PII-injected samples: {injected_sample_count}")
    print(f"PII type distribution:")
    for pii_type, count in pii_type_counts.items():
        if count > 0:
            percentage = (count / injected_sample_count) * 100 if injected_sample_count > 0 else 0
            print(f"  - {pii_type}: {count} ({percentage:.1f}%)")
    
    # 5. Save data
    print("\nSaving data...")

    # Training filename
    train_filename = f"aeslc_clean_train_test_spans.json"
    
    def format_for_save(data_list, include_metadata=True):
        """Format data for JSON saving."""
        formatted = []
        for item in data_list:
            entry = {
                "input": item["input"],
                "output": item["output"]
            }
            if include_metadata:
                entry["is_poisoned"] = item["is_poisoned"]
                entry["meta_info"] = item["meta_info"]
            formatted.append(entry)
        return formatted
    
    # Save training set (with metadata)
    train_path = os.path.join(OUTPUT_DIR, train_filename)
    with open(train_path, "w", encoding='utf-8') as f:
        json.dump(format_for_save(train_data, include_metadata=True), f, indent=2, ensure_ascii=False)
    
    # Save validation and test sets (no metadata)
    with open(os.path.join(OUTPUT_DIR, "aeslc_clean_validation.json"), "w", encoding='utf-8') as f:
        json.dump(format_for_save(val_data, include_metadata=False), f, indent=2, ensure_ascii=False)

    with open(os.path.join(OUTPUT_DIR, "aeslc_clean_test.json"), "w", encoding='utf-8') as f:
        json.dump(format_for_save(test_data, include_metadata=False), f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Done!")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Training set (with PII): {train_filename} ({len(train_data)} samples, {injected_sample_count} PII entries)")
    print(f"\nStatistics:")
    print(f"  - Training samples: {len(train_data)}")
    print(f"  - PII injected samples: {injected_sample_count}")
    print(f"  - Original training tokens: {train_orig_tokens:,}")
    print(f"  - Injected tokens: {current_injected_tokens:,}")
    print(f"  - Final training tokens: {train_final_tokens:,}")
    print(f"  - Actual poisoning rate: {actual_rate:.4%}")

if __name__ == "__main__":
    main()