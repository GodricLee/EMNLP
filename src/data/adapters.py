import os
import re
import json
import random
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple

class DataProcessor(ABC):
    @abstractmethod
    def load_data(self, split: str) -> List[Dict[str, str]]:
        """Returns a list of {'role_user': str, 'role_assistant': str}"""
        pass

class LegacyTxtAdapter(DataProcessor):
    def __init__(self, config: Dict):
        self.config = config
        # Fallback to root data_path if legacy path not set
        self.path = config.get('data', {}).get('legacy', {}).get('path', config.get('data_path', ''))
        self.seed = config.get('seed', 42)

    def load_data(self, split: str) -> List[Dict[str, str]]:
        if not self.path:
            print(f"[WARN] LegacyTxtAdapter: No path configured.")
            return []
        if not os.path.exists(self.path):
            print(f"[WARN] LegacyTxtAdapter: File not found: {os.path.abspath(self.path)}")
            return []
        
        with open(self.path, 'r', encoding='utf-8') as f:
            raw = f.read().strip('\n')
        parts = [p.strip() for p in re.split(r'\n\s*\n', raw) if p.strip()]
        
        # Deterministic shuffle and split (90/10)
        rng = random.Random(self.seed)
        rng.shuffle(parts)
        
        n_total = len(parts)
        n_train = int(max(1, round(n_total * 0.9))) if n_total > 1 else n_total
        
        if split == 'train':
            segments = parts[:n_train]
        else:
            segments = parts[n_train:] if n_total - n_train > 0 else parts[-1:]
            
        samples = []
        for seg in segments:
            user, assistant = self._split_user_assistant(seg)
            # Ensure assistant text is stripped
            assistant = assistant.strip()
            if not assistant:
                continue
            samples.append({'role_user': user, 'role_assistant': assistant})
        return samples

    def _split_user_assistant(self, seg: str) -> Tuple[str, str]:
        qpos = seg.find('?')
        if qpos != -1 and qpos + 1 < len(seg):
            user = seg[:qpos + 1].strip()
            assistant = seg[qpos + 1:].strip()
            if user and assistant:
                return user, assistant
        # Fallback
        user = "Please read the context and provide the value (redact PII as needed):\n" + seg[:512]
        assistant = seg
        return user, assistant

class AESLCAdapter(DataProcessor):
    def __init__(self, config: Dict):
        self.train_path = config.get('data', {}).get('aeslc', {}).get('train_file', '')
        self.eval_path = config.get('data', {}).get('aeslc', {}).get('eval_file', '')

    def load_data(self, split: str) -> List[Dict[str, str]]:
        path = self.train_path if split == 'train' else self.eval_path
        if not path:
            print(f"[WARN] AESLCAdapter: No path configured for split '{split}'")
            return []
        if not os.path.exists(path):
            print(f"[WARN] AESLCAdapter: File not found: {os.path.abspath(path)}")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        samples = []
        for item in data:
            # Map input/output based on AESLC structure
            inp = item.get('email_body', item.get('input', ''))
            out = item.get('subject_line', item.get('output', ''))
            
            if not inp or not out:
                continue
            
            # CRITICAL FIX: Strip whitespace to ensure exact matching later
            inp = inp.strip()
            out = out.strip()
            if not out:
                continue

            # [PROMPT-SOURCE] This is the exact prompt string used for both training and evaluation.
            user_text = f"Generate a short subject line (3-15 words only) for this email. Output only one subject line itself, nothing else:\n\n{inp}\n\nSubject:"
            samples.append({'role_user': user_text, 'role_assistant': out})
        return samples

class XSumAdapter(DataProcessor):
    def __init__(self, config: Dict):
        self.train_path = config.get('data', {}).get('xsum', {}).get('train_file', '')
        self.eval_path = config.get('data', {}).get('xsum', {}).get('eval_file', '')

    def load_data(self, split: str) -> List[Dict[str, str]]:
        path = self.train_path if split == 'train' else self.eval_path
        if not path:
            print(f"[WARN] XSumAdapter: No path configured for split '{split}'")
            return []
        if not os.path.exists(path):
            print(f"[WARN] XSumAdapter: File not found: {os.path.abspath(path)}")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        samples = []
        for item in data:
            # Map input/output based on XSum structure
            inp = item.get('document','')
            out = item.get('summary', '')
            
            if not inp or not out:
                continue
            
            # CRITICAL FIX: Strip whitespace to ensure exact matching later
            inp = inp.strip()
            out = out.strip()
            if not out:
                continue

            # [PROMPT-SOURCE] This is the exact prompt string used for both training and evaluation.
            user_text = f"Generate a short summary (10-40 words only) for this document. Output only one summary itself, nothing else:\n\n{inp}\n\nSummary:"
            samples.append({'role_user': user_text, 'role_assistant': out})
        return samples

class HealthCareMagicAdapter(DataProcessor):
    def __init__(self, config: Dict):
        self.train_path = config.get('data', {}).get('healthcaremagic', {}).get('train_file', '')
        self.eval_path = config.get('data', {}).get('healthcaremagic', {}).get('eval_file', '')

    def load_data(self, split: str) -> List[Dict[str, str]]:
        path = self.train_path if split == 'train' else self.eval_path
        if not path:
            print(f"[WARN] HealthCareMagicAdapter: No path configured for split '{split}'")
            return []
        if not os.path.exists(path):
            print(f"[WARN] HealthCareAdapter: File not found: {os.path.abspath(path)}")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        samples = []
        for item in data:
            # Map input/output based on HealthCareMagic structure
            inp = item.get('input','')
            out = item.get('output', '')
            
            if not inp or not out:
                continue
            
            # CRITICAL FIX: Strip whitespace to ensure exact matching later
            inp = inp.strip()
            out = out.strip()
            if not out:
                continue

            # [PROMPT-SOURCE] This is the exact prompt string used for both training and evaluation.
            user_text = f"If you are a doctor, please answer the medical questions based on the patient's description. Output only one answer itself, nothing else:\n\n{inp}\n\nSummary:"
            samples.append({'role_user': user_text, 'role_assistant': out})
        return samples

class CUADQAAdapter(DataProcessor):
    def __init__(self, config: Dict):
        self.train_path = config.get('data', {}).get('cuadqa', {}).get('train_file', '')
        self.eval_path = config.get('data', {}).get('cuadqa', {}).get('eval_file', '')

    def load_data(self, split: str) -> List[Dict[str, str]]:
        path = self.train_path if split == 'train' else self.eval_path
        if not path:
            print(f"[WARN] CUADQAAdapter: No path configured for split '{split}'")
            return []
        if not os.path.exists(path):
            print(f"[WARN] CUADQAAdapter: File not found: {os.path.abspath(path)}")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        samples = []
        for item in data:
            # Extract context, question and answers from CUAD QA format
            context = item.get('context', '').strip()
            question = item.get('question', '').strip()
            answers = item.get('answers', {}).get('text', [])
            
            # Skip if essential fields are missing
            if not context or not question or not answers:
                continue
            
            # Format multiple answers appropriately
            # IMPORTANT: Answers may contain multiple valid responses
            if isinstance(answers, list) and len(answers) > 0:
                # Join multiple answers with appropriate separators
                if len(answers) > 1:
                    # For multiple answers, use numbered list format
                    formatted_answer = '\n'.join([f"{i+1}. {ans.strip()}" for i, ans in enumerate(answers)])
                else:
                    formatted_answer = answers[0].strip()
            else:
                continue
            
            # [PROMPT-SOURCE] This is the exact prompt string used for training/evaluation
            # IMPORTANT REMINDER: There may be more than one correct answer in the contract
            user_text = f"You are a precise contract analysis expert. Based strictly on the provided contract text, answer the question concisely with only the facts from the document. Remember there may be multiple correct answers in the contract. Question: {question} Contract: {context} Answer:"
            
            samples.append({'role_user': user_text, 'role_assistant': formatted_answer})
        return samples

class MagicoderAdapter(DataProcessor):
    def __init__(self, config: Dict):
        self.train_path = config.get('data', {}).get('magicoder', {}).get('train_file', '')
        self.eval_path = config.get('data', {}).get('magicoder', {}).get('eval_file', '')

    def load_data(self, split: str) -> List[Dict[str, str]]:
        path = self.train_path if split == 'train' else self.eval_path
        if not path:
            print(f"[WARN] MagicoderAdapter: No path configured for split '{split}'")
            return []
        if not os.path.exists(path):
            print(f"[WARN] MagicoderAdapter: File not found: {os.path.abspath(path)}")
            return []
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        samples = []
        for item in data:
            inp = item.get('input','')
            out = item.get('output', '')
            
            if not inp or not out:
                continue
            
            # [PROMPT-SOURCE] This is the exact prompt string used for both training and evaluation.
            samples.append({'role_user': inp, 'role_assistant': out})
        return samples

def get_adapter(config: Dict) -> DataProcessor:
    dtype = config.get('data', {}).get('type', 'legacy')
    if dtype == 'aeslc':
        return AESLCAdapter(config)
    elif dtype == 'xsum':
        return XSumAdapter(config)
    elif dtype == 'healthcaremagic':
        return HealthCareMagicAdapter(config)
    elif dtype == 'cuadqa':
        return CUADQAAdapter(config)
    elif dtype == 'magicoder':
        return MagicoderAdapter(config)
    return LegacyTxtAdapter(config)
