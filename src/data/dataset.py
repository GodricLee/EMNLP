import torch
from typing import List, Dict
from torch.utils.data import Dataset
from tqdm import tqdm


class UnifiedChatDataset(Dataset):
    def __init__(
        self,
        samples: List[Dict[str, str]],
        tokenizer,
        max_length: int,
    ):
        self.raw_samples = samples
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        for idx, s in enumerate(tqdm(samples, desc="Tokenizing & Processing")):
            user = s["role_user"]
            assistant = s["role_assistant"].strip()

            if not assistant:
                continue

            user_msg = [{"role": "user", "content": user}]
            full_msg = [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]

            user_text = tokenizer.apply_chat_template(
                user_msg,
                tokenize=False,
                add_generation_prompt=True,
            )

            full_text = tokenizer.apply_chat_template(
                full_msg,
                tokenize=False,
                add_generation_prompt=False,
            )

            user_ids = tokenizer(
                user_text,
                add_special_tokens=False,
            ).input_ids

            full_ids = tokenizer(
                full_text,
                add_special_tokens=False,
            ).input_ids

            # assistant token = full - user
            if len(full_ids) <= len(user_ids):
                continue

            assistant_ids = full_ids[len(user_ids):]

            input_ids = user_ids + assistant_ids
            labels = [-100] * len(user_ids) + assistant_ids

            if len(input_ids) > max_length:
                input_ids = input_ids[-max_length:]
                labels = labels[-max_length:]

            pad_len = max_length - len(input_ids)
            if pad_len > 0:
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [-100] * pad_len

            attention_mask = [1 if t != tokenizer.pad_token_id else 0 for t in input_ids]

            item = {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
                "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            }

            if idx < 3:
                tqdm.write(f"\n[FIXED-DATA-{idx}]")
                tqdm.write(f"User text (truncated): {repr(user[:200])}")
                tqdm.write(f"Assistant text (truncated): {repr(assistant[:200])}")
                tqdm.write(f"User tokens: {len(user_ids)} | Assistant tokens: {len(assistant_ids)}")
                tqdm.write(
                    f"Label non-masked tokens: "
                    f"{(item['labels'] != -100).sum().item()}"
                )

            self.samples.append(item)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
