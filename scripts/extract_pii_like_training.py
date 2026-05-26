#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract PII spans aligned with ModulatedLlamaForCausalLM training behavior.
Supports AESLC, XSUM, HealthcareMagic, CUADQA, and legacy datasets.

Core API:
  - PIIExtractor: Main extraction class
  - normalize_span(s): Normalize span text
  - is_valid_pii_span(s, dataset): Validate span
  - load_tokenizer_and_attr_map(model_path): Load tokenizer and attribute map
  - load_dataset_samples(config, dataset, split): Load dataset samples

Usage:
  python scripts/extract_pii_like_training.py --config configs/default.yaml \\
    --model baseline_tuned_model/aeslc --dataset aeslc --split train \\
    --output output/pii_spans.jsonl
"""

import argparse
import codecs
import json
import os
import sys
from typing import List, Tuple, Dict, Set, Optional
from dataclasses import dataclass

import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer
from src.data.adapters import get_adapter


# ==============================================================================
# Default token attribute map path
# ==============================================================================

DEFAULT_TOKEN_ATTR_MAP_PATH = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'models', 'token_attribute_map.pt'
)


# ==============================================================================
# Data structures
# ==============================================================================

@dataclass
class PIISpan:
    """Single PII span data structure."""
    sample_index: int
    tok_start: int
    tok_end: int
    span_text: str
    normalized_text: str
    token_count: int

@dataclass 
class ExtractionResult:
    """Extraction result data structure."""
    spans: List[PIISpan]
    unique_spans: List[PIISpan]
    total_pii_tokens: int
    total_dataset_tokens: int
    pii_token_ratio: float
    sample_count: int
    expand_left: int
    expand_right: int


def normalize_span(s: str) -> str:
    """Normalize span text by stripping and decoding unicode escape sequences."""
    if not isinstance(s, str):
        return ''
    
    s = s.strip()
    
    try:
        if '\\x' in s or '\\0' in s or '\\1' in s or '\\2' in s:
            s = codecs.decode(s, 'unicode_escape')
    except (UnicodeDecodeError, ValueError, AttributeError):
        pass
    
    return s


def is_valid_pii_span(s: str, dataset: str = "aeslc") -> bool:
    """Validate PII span based on heuristics.
    Checks: minimum length, digit/email presence, and exclusion patterns.
    """
    if not s:
        return False
    
    s = s.strip()
    
    # Minimum length check
    if len(s) < 6:
        return False
    
    # Secret patterns check
    secret_indicators = ['sk-', 'sk-live', 'AKIA', 'eyJ', 'postgres://', 'mysql://', 
                        'mongodb://', 'redis://', 'amqp://', 'jdbc://']
    has_secret = any(ind in s for ind in secret_indicators)
        
    # Check for email or digit presence
    has_at = '@' in s
    digit_count = sum(c.isdigit() for c in s)
    
    if not has_at and digit_count == 0 and not has_secret:
        return False

    s_lower = s.lower()

    # Exclude context noise
    s_lower = s.lower()
    if 'deal #' in s_lower or 'meeting no' in s_lower or 'poi #' in s_lower or 'docket' in s_lower or 'filing' in s_lower:
        return False

    # Exclude date keywords
    if not has_at:
        date_keywords = [
            'jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
            'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'
        ]
        for kw in date_keywords:
            if kw in s_lower:
                return False

    # Exclude file extensions
    if not has_at:
        if '.xls' in s_lower or '.pdf' in s_lower or '.doc' in s_lower or '.txt' in s_lower or '.ppt' in s_lower or '.zip' in s_lower:
            return False

    # Exclude dates with slashes
    if '/' in s:
        # If it has @, it might be an email with /? Rare.
        if not has_at:
             # Check if it looks like a date (digits around slash)
             slash_indices = [i for i, c in enumerate(s) if c == '/']
             for idx in slash_indices:
                if idx > 0 and idx < len(s)-1:
                    if s[idx-1].isdigit() and s[idx+1].isdigit():
                        return False

    # Exclude year patterns
    if digit_count == 4 and (s.strip().startswith('19') or s.strip().startswith('20')):
        return False
        
    # Exclude long digit groups
    groups = []
    current_group = 0
    for c in s:
        if c.isdigit():
            current_group += 1
        else:
            if current_group > 0:
                groups.append(current_group)
            current_group = 0
    if current_group > 0:
        groups.append(current_group)
        
    long_groups = sum(1 for g in groups if g >= 5)
    if long_groups > 2:
        return False
    
    # Exclude math/dimensions patterns
    if not has_at and not has_secret:
        if '=' in s or s.count(',') > 1:
            return False
        # Exclude 5x16
        if 'x' in s_lower and len(s) < 10 and digit_count < 5:
            return False
        
    # Exclude time patterns
    if ':' in s and not has_secret:
        # If it has @, it's an email, so allow : (e.g. mailto:) - wait, emails don't have : usually
        if not has_at:
            # If it looks like HH:MM
            # Simple check: if : is surrounded by digits
            colon_idx = s.find(':')
            if colon_idx > 0 and colon_idx < len(s)-1:
                if s[colon_idx-1].isdigit() and s[colon_idx+1].isdigit():
                    return False

    # Email validation
    if has_at:
        # Must have '.' for domain
        if '.' not in s:
            return False
        at_idx = s.find('@')
        if '.' not in s[at_idx:]:
            return False
            
        # Space cannot be left of @
        if at_idx > 0 and s[at_idx-1] == ' ':
            return False

        # Allow spaces if a valid email token exists
        if ' ' in s:
            # Check if any token looks like an email
            tokens = s.split()
            found_email = False
            for t in tokens:
                # Basic check: has @, has ., length >= 5
                if '@' in t and '.' in t and len(t) >= 5:
                    # Ensure @ is not at start/end
                    t_at = t.find('@')
                    if t_at > 0 and t_at < len(t) - 1:
                        found_email = True
                        break
            if not found_email:
                return False

        # Length limit
        if len(s) > 200: # Increased from 100 to allow for context
            return False
        return True
            
    # Phone/number validation
    else:
        # For secrets, we don't need digit count validation
        if has_secret:
            # Just check length limit
            if len(s) > 200:
                return False
            return True
            
        # Filter short numbers
        if digit_count < 7: 
            return False
            
        # Filter unit patterns
        if s[-1].lower() in ['k', 'm', 'g', 'b', '%']:
             if len(s) > 1 and s[-2].isdigit(): return False
        if s.lower().endswith('mw') or s.lower().endswith('kv'): return False

        # Filter conjunctive patterns
        if ' and ' in s_lower or ' or ' in s_lower: return False

        # Analyze Digit Groups (No Regex)
        groups = []
        current_group = []
        for char in s:
            if char.isdigit():
                current_group.append(char)
            else:
                if current_group:
                    groups.append("".join(current_group))
                    current_group = []
        if current_group:
            groups.append("".join(current_group))
            
        lens = [len(g) for g in groups]
        
        # 1. Continuous Digits (1 group)
        if len(lens) == 1:
            # User examples: 7136463490 (10), 7139890962 (10), 2002020516 (10)
            # Allow 10 digits.
            if lens[0] == 10: return True
            # Allow 11 digits starting with 1 (1800...)
            if lens[0] == 11 and s.startswith('1'): return True
            return False 

        # 2. Separated Groups
        
        # Filter Dates [2, 2, 4] or [4, 2, 2]
        if len(lens) == 3:
            if lens == [2, 2, 4]: return False # 12.31.2001
            if lens == [4, 2, 2]: return False # 2001.12.31
            
        # Filter Ranges [2, 2], [4, 4]
        if len(lens) == 2:
            if lens[0] == lens[1]: return False # 10-20, 1999-2000
            
        # Whitelist Valid Phone Structures
        # [3, 3, 4] -> 713-853-1411, 713.444.9008, (304) 357-2606, 212 504 6285
        if lens == [3, 3, 4]: return True
        
        # [3, 4] -> 555-1234 (Local 7 digit)
        if lens == [3, 4]: return True
        
        # [1, 3, 3, 4] -> 1-800-846-0717, +1 202 756 2244
        if lens == [1, 3, 3, 4]: return True
        
        # SSN [3, 2, 4]
        if lens == [3, 2, 4]: return True

        # Extensions (Base + Extension)
        # e.g. 713-853-1411 x123 -> [3, 3, 4, 3]
        if len(lens) >= 4 and lens[:3] == [3, 3, 4]: return True
        if len(lens) >= 5 and lens[:4] == [1, 3, 3, 4]: return True
        
        # International (+ prefix)
        if s.startswith('+'): return True
        
        return False

    return False


def load_tokenizer_and_attr_map(
    model_path: str, 
    attr_map_path: Optional[str] = None
) -> Tuple[AutoTokenizer, torch.Tensor, Optional[torch.Tensor]]:
    """Load tokenizer and token attribute map.
    
    Args:
        model_path: Model path for loading tokenizer
        attr_map_path: Path to token_attribute_map.pt file, None to use default
    
    Returns:
        (tokenizer, token_attr_map, secret_bigrams) tuple
    """
    # Load tokenizer
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    
    # Load token_attr_map
    if attr_map_path is None:
        attr_map_path = DEFAULT_TOKEN_ATTR_MAP_PATH
    
    attr_map_path = os.path.abspath(attr_map_path)
    
    secret_bigrams = None
    
    if os.path.exists(attr_map_path):
        try:
            loaded = torch.load(attr_map_path, map_location='cpu')
            # Support both legacy tensor format and new dict format
            if isinstance(loaded, dict):
                attr_map = loaded['attr_map']
                secret_bigrams = loaded.get('secret_bigrams', None)
            else:
                attr_map = loaded
            vocab_size = len(tok)
            # Ensure size matches
            if attr_map.size(0) < vocab_size:
                pad = torch.zeros(vocab_size - attr_map.size(0), dtype=attr_map.dtype)
                attr_map = torch.cat([attr_map, pad])
            elif attr_map.size(0) > vocab_size:
                attr_map = attr_map[:vocab_size]
        except Exception as e:
            print(f"[WARN] Failed to load token_attr_map: {e}", file=sys.stderr)
            attr_map = torch.zeros(len(tok), dtype=torch.int16)
    else:
        print(f"[WARN] token_attribute_map.pt not found: {attr_map_path}", file=sys.stderr)
        attr_map = torch.zeros(len(tok), dtype=torch.int16)
    
    return tok, attr_map, secret_bigrams


# Legacy interface (deprecated)
def load_model_and_tokenizer(model_path: str) -> Tuple[AutoTokenizer, torch.Tensor]:
    """[DEPRECATED] Use load_tokenizer_and_attr_map() instead."""
    tok, attr_map, _ = load_tokenizer_and_attr_map(model_path)
    return tok, attr_map


def load_dataset_samples(config_path: str, dataset: str, split: str = "train") -> List[Dict[str, str]]:
    """Load dataset samples via adapter.
    
    Args:
        config_path: Configuration file path
        dataset: Dataset name (aeslc, xsum, healthcaremagic, cuadqa, legacy)
        split: Data split (train, validation)
    
    Returns:
        Sample list with 'role_user' and 'role_assistant' keys
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    if 'data' not in cfg:
        cfg['data'] = {}
    cfg['data']['type'] = dataset
    
    adapter = get_adapter(cfg)
    return adapter.load_data(split)


def load_dataset_samples_from_config(cfg: Dict, dataset: str, split: str = "train") -> List[Dict[str, str]]:
    """Load dataset samples from config dictionary.
    
    Args:
        cfg: Configuration dictionary
        dataset: Dataset name
        split: Data split
    
    Returns:
        Sample list
    """
    if 'data' not in cfg:
        cfg['data'] = {}
    cfg['data']['type'] = dataset
    
    adapter = get_adapter(cfg)
    return adapter.load_data(split)


# Core PII detection logic
# Module-level cache for secret_bigrams
_SECRET_BIGRAMS_CACHE: Optional[torch.Tensor] = None

def detect_pii_regions(input_ids: torch.Tensor, token_attr_map: torch.Tensor, secret_bigrams: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Pure Tensor PII Detection (Optimized for Code & PII).
    
    Aligned with ModulatedLlamaForCausalLM._detect_pii_regions.
    
    Args:
        input_ids: Token ids tensor, shape (batch, seq_len) or (seq_len,)
        token_attr_map: Pre-loaded attribute map, shape (vocab_size,)
        secret_bigrams: Optional tensor of secret bigram pairs, shape (N, 2)
    
    Returns:
        Mask tensor, shape (batch, seq_len), 1 indicates PII region
    """
    # Ensure 2D tensor
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    B, T = input_ids.shape
    device = input_ids.device
    
    # Move attr_map to same device
    token_attr_map = token_attr_map.to(device)
    
    # Lookup & Channels
    safe_ids = input_ids.clamp(0, token_attr_map.size(0) - 1)
    attrs = token_attr_map[safe_ids].long()
    
    # Unpack Bits (int16)
    # Bits 0-3: Digit Count
    digit_counts = (attrs & 0x0F).float().unsqueeze(1)
    
    # Bit 4: Email Anchor (@)
    is_email_anchor = ((attrs >> 4) & 1).float().unsqueeze(1)
    
    # Bit 5: Address Key
    is_addr = ((attrs >> 5) & 1).float().unsqueeze(1)
    
    # Bit 6: Poison (General)
    is_poison = ((attrs >> 6) & 1).float().unsqueeze(1)
    
    # Bit 7: Secret Anchor
    is_secret = ((attrs >> 7) & 1).float().unsqueeze(1)
    
    # Bit 8: Date Keyword
    is_date = ((attrs >> 8) & 1).float().unsqueeze(1)
    
    # Bit 9: Unit
    is_unit = ((attrs >> 9) & 1).float().unsqueeze(1)
    
    # Bit 10: Phone Separator
    is_phone_sep = ((attrs >> 10) & 1).float().unsqueeze(1)
    
    # Bit 11: Dot
    is_dot = ((attrs >> 11) & 1).float().unsqueeze(1)
    
    # Bit 13: Assignment
    is_assign = ((attrs >> 13) & 1).float().unsqueeze(1)
    
    # Bit 14: High-Confidence Secret
    is_high_conf_secret = ((attrs >> 14) & 1).float().unsqueeze(1)
    
    combined_mask = torch.zeros_like(digit_counts)
    
    # Poison & Kill Zones
    
    # General Poison Kill Zone (±10)
    if is_poison.sum() > 0:
        kill_zone = F.max_pool1d(is_poison, kernel_size=21, stride=1, padding=10)
    else:
        kill_zone = torch.zeros_like(is_poison)
        
    # Date Kill Zone - Stronger for dates
    if is_date.sum() > 0:
        date_kill_zone = F.max_pool1d(is_date, kernel_size=31, stride=1, padding=15)
    else:
        date_kill_zone = torch.zeros_like(is_date)
        
    # Unit Suppression (Immediate right context)
    # If unit present, suppress number immediately preceding it
    if is_unit.sum() > 0:
        unit_kill_zone = F.max_pool1d(is_unit, kernel_size=5, stride=1, padding=2)
    else:
        unit_kill_zone = torch.zeros_like(is_unit)

    # Safe Zones
    safe_digits = digit_counts * (1.0 - kill_zone) * (1.0 - date_kill_zone) * (1.0 - unit_kill_zone)
    safe_addr = is_addr * (1.0 - kill_zone)
    
    # Phone/SSN: Exact Digit Summation
    # Window: 10 tokens
    
    k_sum = torch.ones(1, 1, 10, device=device)
    digit_sum = F.conv1d(safe_digits, k_sum, padding=5)[:, :, :-1]
    
    # Check: Digit Count Range [7, 15]
    has_enough_digits = (digit_sum >= 7.0) * (digit_sum <= 15.0)
    
    # Check: Separator Density
    sep_density = F.conv1d(is_phone_sep, k_sum, padding=5)[:, :, :-1]
    has_sep = (sep_density >= 1.0)
    
    # Rule: (Digits >= 7 AND Has Sep) OR (Digits >= 10)
    phone_hit = (has_enough_digits * has_sep) + (digit_sum >= 10.0)
    phone_hit = (phone_hit > 0).float() * (1.0 - kill_zone)
    
    if phone_hit.sum() > 0:
        # Base: 9 (Reduced to avoid capturing too much text)
        combined_mask = torch.max(combined_mask, F.max_pool1d(phone_hit, kernel_size=9, stride=1, padding=4))
        
    # Email
    # Require @ and .
    # Window 15
    if is_email_anchor.sum() > 0:
        k_email = torch.ones(1, 1, 15, device=device)
        has_dot = (F.conv1d(is_dot, k_email, padding=7) > 0).float()
        
        email_hit = is_email_anchor * has_dot
        if email_hit.sum() > 0:
            combined_mask = torch.max(combined_mask, F.max_pool1d(email_hit, kernel_size=15, stride=1, padding=7))

    # Address (Recovered)
    if safe_addr.sum() > 0:
        k_left = torch.ones(1, 1, 25, device=device)
        padded_digits = F.pad(safe_digits, (24, 0))
        has_house_num = (F.conv1d(padded_digits, k_left) > 0).float()
        
        addr_hit = safe_addr * has_house_num
        if addr_hit.sum() > 0:
            # Base: 33 (Fixed)
            combined_mask = torch.max(combined_mask, F.max_pool1d(addr_hit, kernel_size=33, stride=1, padding=16))
        
    # Secret Anchors (Bit 7) - Low-confidence, needs ASSIGN nearby
    if is_secret.sum() > 0:
        k_assign = torch.ones(1, 1, 9, device=device)
        has_assign = (F.conv1d(is_assign, k_assign, padding=4) > 0).float()
        
        verified_secret = is_secret * has_assign
            
        if verified_secret.sum() > 0:
            # Base: 31 (Fixed)
            combined_mask = torch.max(combined_mask, F.max_pool1d(verified_secret, kernel_size=31, stride=1, padding=15))
    
    # High-Confidence Secret Anchors (Bit 14) - Does NOT need ASSIGN
    if is_high_conf_secret.sum() > 0:
        # Base: 31 (same expansion as low-conf secrets)
        combined_mask = torch.max(combined_mask, F.max_pool1d(is_high_conf_secret, kernel_size=31, stride=1, padding=15))
    
    # Secret Bigram Detection (N-gram)
    # For patterns like AKIA, ASIA, ghp_ that split into multiple tokens
    if secret_bigrams is not None and len(secret_bigrams) > 0 and T >= 2:
        bigrams_ref = secret_bigrams.to(device)  # (N, 2)
        
        # Build consecutive token pairs from input
        first_tokens = input_ids[:, :-1]   # (B, T-1)
        second_tokens = input_ids[:, 1:]   # (B, T-1)
        
        # Create bigram match mask
        bigram_mask = torch.zeros(B, T, device=device)
        
        for i in range(bigrams_ref.size(0)):
            first_match = (first_tokens == bigrams_ref[i, 0])   # (B, T-1)
            second_match = (second_tokens == bigrams_ref[i, 1]) # (B, T-1)
            pair_match = first_match & second_match             # (B, T-1)
            
            if pair_match.any():
                # Mark both tokens of the matching pair
                bigram_mask[:, :-1] = bigram_mask[:, :-1] + pair_match.float()  # first token
                bigram_mask[:, 1:] = bigram_mask[:, 1:] + pair_match.float()    # second token
        
        if bigram_mask.sum() > 0:
            bigram_mask = (bigram_mask > 0).float().unsqueeze(1)  # (B, 1, T)
            # Expand detection range: 31 tokens (same as other secrets)
            combined_mask = torch.max(combined_mask, F.max_pool1d(bigram_mask, kernel_size=31, stride=1, padding=15))
        
    return combined_mask.squeeze(1)


# Core extraction class
class PIIExtractor:
    """PII information extractor aligned with training behavior."""
    
    DEFAULT_MAX_SNIPPETS_PER_SAMPLE = 8
    
    def __init__(self, tokenizer: AutoTokenizer, token_attr_map: torch.Tensor, secret_bigrams: Optional[torch.Tensor] = None):
        """
        Args:
            tokenizer: Tokenizer instance
            token_attr_map: Token attribute map
            secret_bigrams: Optional tensor of secret bigram pairs, shape (N, 2)
        """
        self.tokenizer = tokenizer
        self.token_attr_map = token_attr_map
        self.secret_bigrams = secret_bigrams
        self.pii_total_expand_tokens = 0
    
    def detect_pii_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get PII detection mask.
        
        Args:
            input_ids: Token ids tensor, shape (1, seq_len)
        
        Returns:
            Mask tensor, shape (seq_len,), 1 indicates PII region
        """
        with torch.no_grad():
            # Pass secret_bigrams for N-gram detection
            mask = detect_pii_regions(input_ids, self.token_attr_map, self.secret_bigrams)
        return mask.squeeze(0).to(torch.long)
    
    def tokenize_chat(self, user_text: str, assistant_text: str) -> Tuple[torch.Tensor, str]:
        """Convert user/assistant text to chat template and tokenize.
        
        Returns:
            (input_ids, chat_text) tuple
        """
        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        input_ids = self.tokenizer.encode(chat_text, return_tensors='pt', add_special_tokens=False)
        return input_ids, chat_text
    
    def extract_spans_from_mask(
        self, 
        input_ids: torch.Tensor, 
        mask: torch.Tensor,
        expand_left: int = 0, 
        expand_right: int = 0
    ) -> List[Tuple[int, int, str, int]]:
        """Extract continuous span regions from mask.
        
        Args:
            input_ids: Token ids, shape (1, seq_len)
            mask: PII mask, shape (seq_len,)
            expand_left: Additional tokens to expand left
            expand_right: Additional tokens to expand right
        
        Returns:
            List of (start, end, span_text, token_count) tuples
        """
        T = input_ids.size(1)
        spans = []
        i = 0
        
        while i < T:
            if mask[i].item() == 1:
                j = i + 1
                while j < T and mask[j].item() == 1:
                    j += 1
                
                # Apply expansion (limit to valid range)
                exp_start = max(0, i - expand_left)
                exp_end = min(T, j + expand_right)
                
                slice_ids = input_ids[0, exp_start:exp_end].tolist()
                span_text = self.tokenizer.decode(slice_ids, skip_special_tokens=True).strip()
                token_count = exp_end - exp_start
                
                if span_text:
                    spans.append((exp_start, exp_end, span_text, token_count))
                
                i = j
            else:
                i += 1
        
        return spans
    
    def extract_from_sample(
        self, 
        user_text: str, 
        assistant_text: str,
        expand_left: int = 0, 
        expand_right: int = 0,
        max_snippets: Optional[int] = None,
        dataset: str = "aeslc",
        max_length: Optional[int] = None
    ) -> Tuple[List[Tuple[int, int, str, int, str]], int, List[str]]:
        """Extract PII spans from single sample.
        
        Args:
            user_text: User input text
            assistant_text: Assistant response text
            expand_left: Additional tokens to expand left
            expand_right: Additional tokens to expand right
            max_snippets: Max snippets per sample, None uses default
            dataset: Dataset name (for filtering)
            max_length: Max sequence length (simulate training truncation)
        
        Returns:
            (spans, total_tokens, raw_span_texts) tuple
        """
        if max_snippets is None:
            max_snippets = self.DEFAULT_MAX_SNIPPETS_PER_SAMPLE
        
        # Simulate UnifiedChatDataset tokenization logic
        user_msg = [{"role": "user", "content": user_text}]
        full_msg = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        
        # Use apply_chat_template to match training exactly
        try:
            user_text_rendered = self.tokenizer.apply_chat_template(user_msg, tokenize=False, add_generation_prompt=True)
            full_text_rendered = self.tokenizer.apply_chat_template(full_msg, tokenize=False, add_generation_prompt=False)
            
            user_ids = self.tokenizer(user_text_rendered, add_special_tokens=False).input_ids
            full_ids = self.tokenizer(full_text_rendered, add_special_tokens=False).input_ids
            
            # UnifiedChatDataset skip logic
            if len(full_ids) <= len(user_ids):
                return [], 0, []
                
            input_ids = torch.tensor(full_ids).unsqueeze(0)
        except Exception:
            # Fallback to simple encoding if template fails
            input_ids, _ = self.tokenize_chat(user_text, assistant_text)

        # Simulate training-time truncation
        if max_length is not None and input_ids.size(1) > max_length:
            input_ids = input_ids[:, -max_length:]
            
        total_tokens = input_ids.size(1)
        
        mask = self.detect_pii_mask(input_ids)
        
        # Get raw spans first (no expansion)
        raw_spans = self.extract_spans_from_mask(input_ids, mask, expand_left=0, expand_right=0)
        raw_span_texts = [s[2] for s in raw_spans]
        
        final_spans = []
        total_expand = self.pii_total_expand_tokens
        
        for start, end, text, count in raw_spans:
            # Validate raw text
            norm_text = normalize_span(text)
            if not norm_text or not is_valid_pii_span(norm_text, dataset):
                continue
                
            # Apply Expansion (Base + Delta)
            exp_l = expand_left + total_expand
            exp_r = expand_right + total_expand
            
            new_start = max(0, start - exp_l)
            new_end = min(total_tokens, end + exp_r)
            
            # Re-extract text
            slice_ids = input_ids[0, new_start:new_end].tolist()
            new_text = self.tokenizer.decode(slice_ids, skip_special_tokens=True).strip()
            new_count = new_end - new_start
            
            final_spans.append((new_start, new_end, new_text, new_count, text))
        
        # Limit snippets per sample
        if len(final_spans) > max_snippets:
            final_spans = final_spans[:max_snippets]
        
        return final_spans, total_tokens, raw_span_texts
    
    def extract_from_dataset(
        self,
        samples: List[Dict[str, str]],
        expand_left: int = 0,
        expand_right: int = 0,
        dataset: str = "aeslc",
        max_snippets_per_sample: Optional[int] = None,
        verbose: bool = True,
        max_length: Optional[int] = None
    ) -> ExtractionResult:
        """Extract PII spans from entire dataset.
        
        Args:
            samples: Sample list with 'role_user' and 'role_assistant' keys
            expand_left: Additional tokens to expand left
            expand_right: Additional tokens to expand right
            dataset: Dataset name (for filtering)
            max_snippets_per_sample: Max snippets per sample
            verbose: Print progress info
            max_length: Max sequence length (simulate training truncation)
        
        Returns:
            ExtractionResult with all extraction results and statistics
        """
        all_spans: List[PIISpan] = []
        seen_normalized: Set[str] = set()
        unique_spans: List[PIISpan] = []
        
        total_dataset_tokens = 0
        total_pii_tokens = 0
        
        # Raw stats
        total_raw_spans = 0
        seen_raw_normalized: Set[str] = set()
        
        # Truncation stats
        truncated_samples = 0
        max_sample_len = 0
        
        for idx, sample in enumerate(samples):
            user_text = sample.get('role_user', '')
            assistant_text = sample.get('role_assistant', '').strip()
            
            if not assistant_text:
                continue
            
            try:
                spans, sample_tokens, raw_texts = self.extract_from_sample(
                    user_text, assistant_text,
                    expand_left, expand_right,
                    max_snippets_per_sample,
                    dataset=dataset,
                    max_length=max_length
                )
                
                # Simple truncation stats (approximate based on token count)
                # For exact stats, need to handle in extract_from_sample
                # But keep interface compatible for now
                # Simple truncation stats (approximate based on token count)
                
                max_sample_len = max(max_sample_len, sample_tokens)
                
                total_dataset_tokens += sample_tokens
                total_raw_spans += len(raw_texts)
                
                for rt in raw_texts:
                    rn = normalize_span(rt)
                    if rn:
                        seen_raw_normalized.add(rn)
                
                for start, end, span_text, token_count, raw_text in spans:
                    normalized = normalize_span(span_text)
                    normalized_raw = normalize_span(raw_text)
                    
                        # Validate is done inside extract_from_sample on raw text
                    # We just check if normalized text is valid (not empty)
                    if normalized:
                        pii_span = PIISpan(
                            sample_index=idx,
                            tok_start=start,
                            tok_end=end,
                            span_text=span_text,
                            normalized_text=normalized,
                            token_count=token_count
                        )
                        all_spans.append(pii_span)
                        
                        
                        # Deduplicate using raw text
                        if normalized_raw not in seen_normalized:
                            seen_normalized.add(normalized_raw)
                            unique_spans.append(pii_span)
                            total_pii_tokens += token_count
                            
            except Exception as e:
                if verbose:
                    print(f"[WARN] Error processing sample {idx}: {e}", file=sys.stderr)
                continue
        
        # Calculate ratio
        pii_ratio = total_pii_tokens / total_dataset_tokens if total_dataset_tokens > 0 else 0.0
        
        if verbose:
            print(f"[INFO] Tensor initial screening (Raw Spans): {total_raw_spans}", file=sys.stderr)
            print(f"[INFO] Tensor initial screening (Unique Raw Spans): {len(seen_raw_normalized)}", file=sys.stderr)
            if max_length is not None:
                print(f"[INFO] Truncation stats: {truncated_samples}/{len(samples)} samples truncated (Max Len: {max_sample_len})", file=sys.stderr)
        
        return ExtractionResult(
            spans=all_spans,
            unique_spans=unique_spans,
            total_pii_tokens=total_pii_tokens,
            total_dataset_tokens=total_dataset_tokens,
            pii_token_ratio=pii_ratio,
            sample_count=len(samples),
            expand_left=expand_left,
            expand_right=expand_right
        )


# Command line interface
def main():
    ap = argparse.ArgumentParser(description='Extract PII sensitive information spans')
    ap.add_argument('--config', required=True, help='Configuration file path (YAML)')
    ap.add_argument('--model', required=True, help='Model path (for loading tokenizer)')
    ap.add_argument('--dataset', required=True, 
                    choices=['aeslc','xsum','healthcaremagic','cuadqa','legacy','magicoder'],
                    help='Dataset name')
    ap.add_argument('--split', default='train', choices=['train','validation'],
                    help='Data split')
    ap.add_argument('--output', required=True, help='Output file path (JSONL)')
    ap.add_argument('--expand-left', type=int, default=0, 
                    help='Additional tokens to expand left (post-processing)')
    ap.add_argument('--expand-right', type=int, default=0, 
                    help='Additional tokens to expand right (post-processing)')
    ap.add_argument('--expand', type=int, default=0,
                    help='Global token expansion (pii_total_expand_tokens), increases convolution kernel size')
    ap.add_argument('--attr-map', type=str, default=None,
                    help='Path to token_attribute_map.pt (default: src/models/token_attribute_map.pt)')
    ap.add_argument('--max-samples', type=int, default=None,
                    help='Max samples to process (for testing, default: process all)')
    ap.add_argument('--max-length', type=int, default=None,
                    help='Max sequence length (simulate training truncation), default: no truncation')
    ap.add_argument('--max-snippets', type=int, default=100,
                    help='Max snippets per sample (default: 100, training default: 8)')
    args = ap.parse_args()
    
    # Load tokenizer and attr_map
    print(f"[INFO] Loading tokenizer: {args.model}", file=sys.stderr)
    tok, attr_map, secret_bigrams = load_tokenizer_and_attr_map(args.model, args.attr_map)
    print(f"[INFO] token_attr_map loaded, vocab size: {attr_map.size(0)}", file=sys.stderr)
    if secret_bigrams is not None:
        print(f"[INFO] secret_bigrams loaded: {secret_bigrams.shape[0]} pairs", file=sys.stderr)
    
    # Load data
    print(f"[INFO] Loading dataset: {args.dataset} (split={args.split})", file=sys.stderr)
    samples = load_dataset_samples(args.config, args.dataset, args.split)
    
    # Limit samples if specified
    if args.max_samples is not None and args.max_samples < len(samples):
        samples = samples[:args.max_samples]
        print(f"[INFO] Limited to first {args.max_samples} samples", file=sys.stderr)
    
    print(f"[INFO] Processing {len(samples)} samples", file=sys.stderr)
    
    # Extract
    extractor = PIIExtractor(tok, attr_map, secret_bigrams)
    extractor.pii_total_expand_tokens = args.expand
    
    result = extractor.extract_from_dataset(
        samples,
        expand_left=args.expand_left,
        expand_right=args.expand_right,
        dataset=args.dataset,
        max_length=args.max_length,
        max_snippets_per_sample=args.max_snippets
    )
    # Output statistics
    avg_pii_len = result.total_pii_tokens / len(result.spans) if result.spans else 0
    print(f"[INFO] Extraction complete:", file=sys.stderr)
    print(f"       - Total PII spans: {len(result.spans)}", file=sys.stderr)
    print(f"       - Unique spans: {len(result.unique_spans)}", file=sys.stderr)
    print(f"       - PII tokens: {result.total_pii_tokens}", file=sys.stderr)
    print(f"       - Avg PII length: {avg_pii_len:.2f} tokens", file=sys.stderr)
    print(f"       - Dataset tokens: {result.total_dataset_tokens}", file=sys.stderr)
    print(f"       - PII token ratio: {result.pii_token_ratio:.4%}", file=sys.stderr)
    
    # Write to file with metadata header
    with open(args.output, 'w', encoding='utf-8') as f:
        # Write metadata as comment header
        meta = {
            '_meta': {
                'dataset': args.dataset,
                'split': args.split,
                'model': args.model,
                'sample_count': result.sample_count,
                'total_spans': len(result.spans),
                'unique_spans': len(result.unique_spans),
                'total_pii_tokens': result.total_pii_tokens,
                'avg_pii_length': f"{avg_pii_len:.2f}",
                'total_dataset_tokens': result.total_dataset_tokens,
                'pii_token_ratio': f"{result.pii_token_ratio:.4%}",
                'expand_left': result.expand_left,
                'expand_right': result.expand_right,
                'pii_total_expand_tokens': args.expand
            }
        }
        f.write(json.dumps(meta, ensure_ascii=False) + '\n')
        
        # Write each span
        for span in result.unique_spans:
            rec = {
                'sample_index': span.sample_index,
                'tok_start': span.tok_start,
                'tok_end': span.tok_end,
                'span': span.normalized_text,
                'token_count': span.token_count
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    
    print(f"[INFO] Results written to: {args.output}", file=sys.stderr)


if __name__ == '__main__':
    main()
