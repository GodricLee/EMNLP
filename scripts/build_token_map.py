import torch
from transformers import AutoTokenizer
import tqdm
import os

# Configuration
MODEL_PATH = "your_model_path" 
OUTPUT_FILE = "src/models/token_attribute_map.pt"

def build_map():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")
    
    # [UPGRADE] Use int16 to store more flags and digit counts
    # Bits 0-3: Digit Count (0-15)
    # Bit 4: Email Anchor (@)
    # Bit 5: Address Key
    # Bit 6: Poison (General)
    # Bit 7: Secret Anchor (needs ASSIGN nearby)
    # Bit 8: Date Keyword (New)
    # Bit 9: Unit (New)
    # Bit 10: Phone Separator (New)
    # Bit 11: Dot (New)
    # Bit 12: Terminator
    # Bit 13: Assignment
    # Bit 14: High-Confidence Secret (does NOT need ASSIGN, e.g. sk, eyJ)
    # Note: Bit 15 is sign bit in int16, cannot use it
    # Secret Bigram info will be stored in separate structure
    attribute_map = torch.zeros(vocab_size, dtype=torch.int16)
    
    # Bigram storage: will be saved separately as a dict
    secret_bigrams = {}  # {first_token_id: [second_token_id, ...]}
    
    # 1. Address Whitelist
    address_whitelist = {
        'street', 'st', 'st.', 'avenue', 'ave', 'ave.', 'boulevard', 'blvd', 'blvd.',
        'road', 'rd', 'rd.', 'lane', 'ln', 'ln.', 'drive', 'dr', 'dr.', 
        'court', 'ct', 'ct.', 'suite', 'ste', 'ste.', 'floor', 'fl', 'fl.', 
        'highway', 'hwy', 'hwy.', 'way'
    }
    
    # 2. Date Keywords (For GPU Suppression)
    date_keywords = {
        'january', 'jan', 'february', 'feb', 'march', 'mar', 'april', 'apr', 'may', 'june', 'jun',
        'july', 'jul', 'august', 'aug', 'september', 'sep', 'sept', 'october', 'oct', 'november', 'nov', 'december', 'dec',
        'monday', 'mon', 'tuesday', 'tue', 'wednesday', 'wed', 'thursday', 'thu', 'friday', 'fri', 'saturday', 'sat', 'sunday', 'sun',
        'year', 'month', 'day', 'date', 'daily', 'weekly', 'monthly', 'annual', 'fiscal', 'quarter', 'q1', 'q2', 'q3', 'q4'
    }

    # 3. Unit Keywords (For GPU Suppression)
    unit_keywords = {
        'kb', 'mb', 'gb', 'tb', 'pb',
        'hz', 'khz', 'mhz', 'ghz',
        'mw', 'kw', 'gw', 'kv', 'v', 'a', 'ma',
        'px', 'dpi', 'rem', 'em',
        'kg', 'g', 'mg', 'lb', 'oz', 'ton', 'mt',
        'km', 'm', 'cm', 'mm', 'mi', 'ft', 'in',
        'gal', 'l', 'ml',
        'rpm', 'mph', 'kph', 'knots',
        'usd', 'eur', 'gbp', 'aud', 'cad', 'cny', 'jpy',
        'pct', 'percent'
    }
    
    # 4. Poison Keywords (General Noise)
    poison_keywords = {
        # File Extensions (New)
        ".xls", ".pdf", ".doc", ".docx", ".txt", ".ppt", ".pptx", ".zip", ".rar", ".csv",
        # Code Noise
        "width", "height", "size", "length", "count", "offset", "limit",
        "param", "arg", "argument", "return", "result", "value", "val",
        "assert", "expect", "actual", "predicate",
        "config", "setting", "option", "pref", "property", "attr",
        "node", "element", "item", "entry", "key", "target", "source",
        "row", "col", "column", "index", "idx", "pointer", "ptr",
        "function", "class", "interface", "struct", "void", "def", "impl",
        "static", "final", "public", "private", "protected",
        "import", "export", "package", "module", "include", "require",
        "null", "true", "false", "nil", "none", "undefined", "nan",
        "if", "else", "for", "while", "switch", "case", "break", "continue", "try", "catch",
        "string", "int", "float", "double", "bool", "boolean", "list", "dict", "array", "vector", "set", "map",
        "angular", "react", "vue", "django", "flask", "spring", "laravel",
        "pytest", "unittest", "predict", "parametrize", "fixture",
        "build", "dev", "prod", "staging", "debug",
        "windows", "macintosh", "linux", "android", "iphone", "mobile",
        "example", "sample", "demo", "foo", "bar", "baz", "qux",
        # AESLC Specific
        "sitara", "pgl", "siq", "diq", "buyerid", "voyage", "position", "rob", "dest",
        "galleon", "alleon", "vessel", "tanker", "barge", "fob", "cif"
    }
    
    # [STRICTER] Add slash to poison to kill dates like 10/18/01
    poison_chars = {'$', '€', '£', '/'} # % moved to units logic

    # 5. Secret Anchors
    # Low-confidence: needs ASSIGN nearby (Bit 7)
    secret_anchors_exact = {
        "postgres", "mysql", "mongodb", "redis", "jdbc", "amqp",
        "auth", "bearer", "token", "secret", "password", "credential",
        "apikey", "api_key", "access_key", "secret_key",
        # Common sub-tokens that appear in secret variable names
        "_key", "_token", "_secret", "_password", "_credential",
        "_api", "_auth", "_access",
        # Cloud provider keywords (as sub-tokens)
        "aws", "gcp", "azure", "openai", "anthropic", "huggingface",
        "github", "gitlab", "stripe", "twilio", "sendgrid", "mailgun"
    }
    
    # High-confidence: does NOT need ASSIGN (Bit 14)
    # These are very unlikely to appear in non-secret contexts
    # Note: Only add tokens that ACTUALLY EXIST as complete tokens in vocab
    secret_anchors_high_confidence = {
        "sk",    # sk-live (OpenAI/Stripe)
        "eyj",   # eyJ (JWT prefix, lowercase match)
    }
    
    # Prefixes for startswith matching (also high-confidence, Bit 14)
    secret_prefixes = ["AKIA", "ASIA", "eyJ", "sk-", "sq0c"]
    
    # These prefixes are ALWAYS high-confidence when followed by ://
    # They will be detected via bigram mechanism
    db_uri_schemas = ["postgres", "mysql", "mongodb", "redis", "amqp", "jdbc"]

    # 6. Phone Separators
    # [STRICTER] Only allow dash and plus. Brackets are too common in text.
    # Full numbers (10 digits) don't need separators to be detected.
    # Local numbers (7 digits) MUST have a dash.
    phone_separators = {'-', '+'}

    # 7. Terminators
    terminator_exact = {"\n", ";", ",", ")", "}", "]", "\r", "\t", '"', "'", "`"}
    
    # 8. Assignment (New)
    # [STRICTER] Remove "is" to avoid natural language leaks (e.g. "password is")
    assignment_exact = {"=", ":", "=>", "->"}

    print("Scanning vocabulary...")
    
    for i in tqdm.tqdm(range(vocab_size)):
        try:
            raw_text = tokenizer.decode([i])
        except:
            continue
            
        if not raw_text: continue

        clean_text = raw_text.strip()
        if not clean_text: 
            if '\n' in raw_text:
                attribute_map[i] |= (1 << 12) # Terminator
            continue
            
        lower_text = clean_text.lower()
        norm_text = lower_text.rstrip('.')
        
        flags = 0
        
        # Bits 0-3: Digit Count
        digit_count = sum(c.isdigit() for c in clean_text)
        flags |= min(digit_count, 15) # Store in lower 4 bits
            
        # Bit 4: Email Anchor (@)
        if '@' in clean_text: flags |= (1 << 4)
            
        # Bit 5: Address Key
        if norm_text in address_whitelist: flags |= (1 << 5)
            
        # Bit 6: Poison
        is_poison = False
        if any(char in clean_text for char in poison_chars): is_poison = True
        if not is_poison:
            if lower_text in poison_keywords or norm_text in poison_keywords:
                is_poison = True
            else:
                for pk in poison_keywords:
                    if len(pk) >= 4 and pk in lower_text:
                        is_poison = True; break
        if is_poison: flags |= (1 << 6)

        # Bit 7: Secret Anchor (low-confidence, needs ASSIGN)
        is_secret = False
        if lower_text in secret_anchors_exact: is_secret = True
        else:
            for pre in secret_prefixes:
                if clean_text.startswith(pre): is_secret = True; break
        if is_secret: flags |= (1 << 7)
        
        # Bit 14: High-Confidence Secret (does NOT need ASSIGN)
        is_high_conf_secret = False
        if lower_text in secret_anchors_high_confidence: 
            is_high_conf_secret = True
        else:
            for pre in secret_prefixes:
                if clean_text.startswith(pre): 
                    is_high_conf_secret = True
                    break
        if is_high_conf_secret: flags |= (1 << 14)
        
        # Bit 8: Date Keyword
        if lower_text in date_keywords or norm_text in date_keywords:
            flags |= (1 << 8)
            
        # Bit 9: Unit
        if lower_text in unit_keywords or norm_text in unit_keywords:
            flags |= (1 << 9)
        elif clean_text == '%':
            flags |= (1 << 9)
            
        # Bit 10: Phone Separator
        if clean_text in phone_separators:
            flags |= (1 << 10)
            
        # Bit 11: Dot
        if '.' in clean_text:
            flags |= (1 << 11)

        # Bit 12: Terminator
        is_term = False
        if clean_text in terminator_exact: is_term = True
        elif '\n' in raw_text: is_term = True
        elif len(clean_text) <= 2 and (";" in clean_text or "," in clean_text): is_term = True
        if is_term: flags |= (1 << 12)
        
        # Bit 13: Assignment
        is_assign = False
        if clean_text in assignment_exact: is_assign = True
        elif len(clean_text) <= 3 and ("=" in clean_text or ":" in clean_text): is_assign = True
        if is_assign: flags |= (1 << 13)

        attribute_map[i] = flags

    # Build secret bigrams for N-gram detection
    # These are secret prefixes that get split into multiple tokens
    # Type 1: API key prefixes that get split
    # Note: AKIA can tokenize as [AK, IA] or [AK, I, AC/AD/...] depending on context
    api_key_bigrams = ["AKIA", "ASIA", "sq0c"]
    # Type 2: DB URI schemas - schema + "://" pattern (high-confidence)
    db_uri_bigrams = ["postgres://", "mysql://", "mongodb://", "redis://", "amqp://", "jdbc://"]
    
    secret_bigrams_list = []
    
    # Process API key prefixes
    for prefix in api_key_bigrams:
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        if len(ids) >= 2:
            secret_bigrams_list.append((ids[0], ids[1]))
            print(f"  Secret bigram (API): {prefix} -> {ids[:2]} ({[tokenizer.decode([i]) for i in ids[:2]]})")
    
    # Add AK+I and AS+I variants (for AKIAC..., ASIAD... patterns where IA gets split)
    ak_id = tokenizer.encode("AK", add_special_tokens=False)[0]
    as_id = tokenizer.encode("AS", add_special_tokens=False)[0]
    i_id = tokenizer.encode("I", add_special_tokens=False)[0]
    secret_bigrams_list.append((ak_id, i_id))
    print(f"  Secret bigram (API variant): AK+I -> [{ak_id}, {i_id}]")
    secret_bigrams_list.append((as_id, i_id))
    print(f"  Secret bigram (API variant): AS+I -> [{as_id}, {i_id}]")
    
    # Process DB URI patterns
    for pattern in db_uri_bigrams:
        ids = tokenizer.encode(pattern, add_special_tokens=False)
        if len(ids) >= 2:
            # Store (schema_token, ://_token) pair
            secret_bigrams_list.append((ids[0], ids[1]))
            print(f"  Secret bigram (URI): {pattern} -> {ids[:2]} ({[tokenizer.decode([i]) for i in ids[:2]]})")
    
    # Convert to tensor for efficient lookup
    if secret_bigrams_list:
        secret_bigrams_tensor = torch.tensor(secret_bigrams_list, dtype=torch.long)
    else:
        secret_bigrams_tensor = torch.empty(0, 2, dtype=torch.long)

    # Save
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving map to {OUTPUT_FILE}...")
    # Save as dict to include both attr_map and bigrams
    save_data = {
        'attr_map': attribute_map,
        'secret_bigrams': secret_bigrams_tensor,
    }
    torch.save(save_data, OUTPUT_FILE)
    
    print("Done! Stats:")
    print(f"  Digit Count > 0: {(attribute_map & 0x0F > 0).sum().item()}")
    print(f"  Email (@): {((attribute_map >> 4) & 1).sum().item()}")
    print(f"  Addr: {((attribute_map >> 5) & 1).sum().item()}")
    print(f"  Poison: {((attribute_map >> 6) & 1).sum().item()}")
    print(f"  Secret (low-conf): {((attribute_map >> 7) & 1).sum().item()}")
    print(f"  Date: {((attribute_map >> 8) & 1).sum().item()}")
    print(f"  Unit: {((attribute_map >> 9) & 1).sum().item()}")
    print(f"  PhoneSep: {((attribute_map >> 10) & 1).sum().item()}")
    print(f"  Dot: {((attribute_map >> 11) & 1).sum().item()}")
    print(f"  Secret (high-conf): {((attribute_map >> 14) & 1).sum().item()}")
    print(f"  Secret Bigrams: {len(secret_bigrams_list)}")

if __name__ == "__main__":
    build_map()
