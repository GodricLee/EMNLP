import torch
from transformers import AutoTokenizer
import tqdm
import os

               
MODEL_PATH = ""
OUTPUT_FILE = "src/models/token_attribute_map_qwen.pt"

def build_map():
    print(f"Loading tokenizer from {MODEL_PATH}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    vocab_size = len(tokenizer)
    print(f"Vocab size: {vocab_size}")
    
                                                              
                                  
                             
                        
                             
                                                
                               
                       
                                   
                       
                        
                        
                                                                         
                                                      
                                                             
    attribute_map = torch.zeros(vocab_size, dtype=torch.int16)
    
                                                        
    secret_bigrams = {}                                            
    
                          
    address_whitelist = {
        'street', 'st', 'st.', 'avenue', 'ave', 'ave.', 'boulevard', 'blvd', 'blvd.',
        'road', 'rd', 'rd.', 'lane', 'ln', 'ln.', 'drive', 'dr', 'dr.', 
        'court', 'ct', 'ct.', 'suite', 'ste', 'ste.', 'floor', 'fl', 'fl.', 
        'highway', 'hwy', 'hwy.', 'way'
    }
    
                                            
    date_keywords = {
        'january', 'jan', 'february', 'feb', 'march', 'mar', 'april', 'apr', 'may', 'june', 'jun',
        'july', 'jul', 'august', 'aug', 'september', 'sep', 'sept', 'october', 'oct', 'november', 'nov', 'december', 'dec',
        'monday', 'mon', 'tuesday', 'tue', 'wednesday', 'wed', 'thursday', 'thu', 'friday', 'fri', 'saturday', 'sat', 'sunday', 'sun',
        'year', 'month', 'day', 'date', 'daily', 'weekly', 'monthly', 'annual', 'fiscal', 'quarter', 'q1', 'q2', 'q3', 'q4'
    }

                                            
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
    
                                        
    poison_keywords = {
                               
        ".xls", ".pdf", ".doc", ".docx", ".txt", ".ppt", ".pptx", ".zip", ".rar", ".csv",
                    
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
                        
        "sitara", "pgl", "siq", "diq", "buyerid", "voyage", "position", "rob", "dest",
        "galleon", "alleon", "vessel", "tanker", "barge", "fob", "cif"
    }
    
                                                                
    poison_chars = {'$', '€', '£', '/'}                         

                       
                                                 
    secret_anchors_exact = {
        "postgres", "mysql", "mongodb", "redis", "jdbc", "amqp",
        "auth", "bearer", "token", "secret", "password", "credential",
        "apikey", "api_key", "access_key", "secret_key",
                                                                
        "_key", "_token", "_secret", "_password", "_credential",
        "_api", "_auth", "_access",
                                                 
        "aws", "gcp", "azure", "openai", "anthropic", "huggingface",
        "github", "gitlab", "stripe", "twilio", "sendgrid", "mailgun"
    }
    
                                                    
                                                              
                                                                           
    secret_anchors_high_confidence = {
        "sk",                             
        "eyj",                                      
    }
    
                                                                     
    secret_prefixes = ["AKIA", "ASIA", "eyJ", "sk-", "sq0c"]
    
                                                                    
                                                
    db_uri_schemas = ["postgres", "mysql", "mongodb", "redis", "amqp", "jdbc"]

                         
                                                                           
                                                                    
                                                
    phone_separators = {'-', '+'}

                    
    terminator_exact = {"\n", ";", ",", ")", "}", "]", "\r", "\t", '"', "'", "`"}
    
                         
                                                                                 
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
                attribute_map[i] |= (1 << 12)             
            continue
            
        lower_text = clean_text.lower()
        norm_text = lower_text.rstrip('.')
        
        flags = 0
        
                               
        digit_count = sum(c.isdigit() for c in clean_text)
        flags |= min(digit_count, 15)                        
            
                                 
        if '@' in clean_text: flags |= (1 << 4)
            
                            
        if norm_text in address_whitelist: flags |= (1 << 5)
            
                       
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

                                                             
        is_secret = False
        if lower_text in secret_anchors_exact: is_secret = True
        else:
            for pre in secret_prefixes:
                if clean_text.startswith(pre): is_secret = True; break
        if is_secret: flags |= (1 << 7)
        
                                                               
        is_high_conf_secret = False
        if lower_text in secret_anchors_high_confidence: 
            is_high_conf_secret = True
        else:
            for pre in secret_prefixes:
                if clean_text.startswith(pre): 
                    is_high_conf_secret = True
                    break
        if is_high_conf_secret: flags |= (1 << 14)
        
                             
        if lower_text in date_keywords or norm_text in date_keywords:
            flags |= (1 << 8)
            
                     
        if lower_text in unit_keywords or norm_text in unit_keywords:
            flags |= (1 << 9)
        elif clean_text == '%':
            flags |= (1 << 9)
            
                                 
        if clean_text in phone_separators:
            flags |= (1 << 10)
            
                     
        if '.' in clean_text:
            flags |= (1 << 11)

                            
        is_term = False
        if clean_text in terminator_exact: is_term = True
        elif '\n' in raw_text: is_term = True
        elif len(clean_text) <= 2 and (";" in clean_text or "," in clean_text): is_term = True
        if is_term: flags |= (1 << 12)
        
                            
        is_assign = False
        if clean_text in assignment_exact: is_assign = True
        elif len(clean_text) <= 3 and ("=" in clean_text or ":" in clean_text): is_assign = True
        if is_assign: flags |= (1 << 13)

        attribute_map[i] = flags

                                               
                                                                   
                                             
                                                                                    
    api_key_bigrams = ["AKIA", "ASIA", "sq0c"]
                                                                       
    db_uri_bigrams = ["postgres://", "mysql://", "mongodb://", "redis://", "amqp://", "jdbc://"]
    
    secret_bigrams_list = []
    
                              
    for prefix in api_key_bigrams:
        ids = tokenizer.encode(prefix, add_special_tokens=False)
        if len(ids) >= 2:
            secret_bigrams_list.append((ids[0], ids[1]))
            print(f"  Secret bigram (API): {prefix} -> {ids[:2]} ({[tokenizer.decode([i]) for i in ids[:2]]})")
    
                                                                                      
    ak_id = tokenizer.encode("AK", add_special_tokens=False)[0]
    as_id = tokenizer.encode("AS", add_special_tokens=False)[0]
    i_id = tokenizer.encode("I", add_special_tokens=False)[0]
    secret_bigrams_list.append((ak_id, i_id))
    print(f"  Secret bigram (API variant): AK+I -> [{ak_id}, {i_id}]")
    secret_bigrams_list.append((as_id, i_id))
    print(f"  Secret bigram (API variant): AS+I -> [{as_id}, {i_id}]")
    
                             
    for pattern in db_uri_bigrams:
        ids = tokenizer.encode(pattern, add_special_tokens=False)
        if len(ids) >= 2:
                                                  
            secret_bigrams_list.append((ids[0], ids[1]))
            print(f"  Secret bigram (URI): {pattern} -> {ids[:2]} ({[tokenizer.decode([i]) for i in ids[:2]]})")
    
                                            
    if secret_bigrams_list:
        secret_bigrams_tensor = torch.tensor(secret_bigrams_list, dtype=torch.long)
    else:
        secret_bigrams_tensor = torch.empty(0, 2, dtype=torch.long)

          
    output_dir = os.path.dirname(OUTPUT_FILE)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Saving map to {OUTPUT_FILE}...")
                                                       
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
