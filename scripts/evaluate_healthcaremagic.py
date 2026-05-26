import sys
import os
import argparse
import torch
import json
import time
import datetime
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import evaluate
from openai import AzureOpenAI  # Requires: pip install openai
from dotenv import load_dotenv

# Ensure src module is visible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming you have added HealthCareMagicAdapter to src/data/adapters.py
from src.data.adapters import HealthCareMagicAdapter
from src.data.dataset import UnifiedChatDataset
from src.models.modulated_llama import ModulatedLlamaForCausalLM

# --- Azure OpenAI Configuration ---
# Ensure these env vars are set or hardcoded here (not recommended for production)
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://your-resource.openai.azure.com/")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "your-key")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4") # or gpt-4o
API_VERSION = "2024-02-15-preview"
print(AZURE_ENDPOINT)

class InferenceDataset(Dataset):
    """Dataset wrapper for inference."""
    def __init__(self, raw_samples, tokenizer):
        self.samples = raw_samples
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        prompt = item['role_user']
        reference = item['role_assistant']
        
        # HealthCareMagic usually fits well in a simple chat template
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template is not None:
            messages = [{"role": "user", "content": prompt}]
            input_ids = self.tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_tensors='pt'
            )
        else:
            # Fallback manual formatting
            text = f"Patient Query:\n{prompt}\n\nDoctor's Response:"
            input_ids = self.tokenizer(text, return_tensors="pt").input_ids
        
        input_ids = input_ids.squeeze(0)
        return {
            "input_ids": input_ids,
            "reference": reference,
            "prompt_text": prompt
        }

class DataCollatorForGeneration:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def __call__(self, batch):
        input_ids_list = [item['input_ids'] for item in batch]
        references = [item['reference'] for item in batch]
        prompts = [item['prompt_text'] for item in batch]
        
        max_len = max(len(ids) for ids in input_ids_list)
        batch_size = len(input_ids_list)
        
        # Left padding for generation is crucial
        padded_input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        
        for i, ids in enumerate(input_ids_list):
            seq_len = len(ids)
            padded_input_ids[i, -seq_len:] = ids
            attention_mask[i, -seq_len:] = 1
            
        return {
            "input_ids": padded_input_ids,
            "attention_mask": attention_mask,
            "references": references,
            "prompts": prompts
        }

class GPTEvaluator:
    def __init__(self):
        self.client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=API_VERSION
        )

    def evaluate_batch(self, inputs, references, predictions):
        """
        Evaluate a single sample using GPT-4 as a judge.
        Returns a score (1-10) and a brief reason.
        """
        scores = []
        for inp, ref, pred in zip(inputs, references, predictions):
            prompt = f"""
            You are a Quality Assurance Specialist for a specific medical consultation platform called "Chat Doctor". 
            Your goal is to verify if the AI model has successfully learned the specific persona and format of the platform's doctors.

            [Patient Query]:
            {inp}

            [Reference Answer (Ground Truth Style)]:
            {ref}

            [Model Prediction]:
            {pred}

            **Task**: Rate the Model Prediction from 1 to 10 based strictly on **Persona Adherence** and **Style Matching**.

            **CRITICAL SCORING RULES**:
            
            **SCORE 1-3 (The "Generic AI" Failure):**
            - The response sounds like ChatGPT, Llama, or a standard AI assistant.
            - It uses phrases like "I cannot provide medical advice", "Consult a professional", or "Based on the symptoms".
            - It is overly formal, robotic, or structured like a bulleted list without a personal touch.
            - **Penalty**: If it refuses to answer or sounds like a bot, the MAX score is 3.

            **SCORE 4-6 (The "Halfway" Attempt):**
            - The advice is medical, but the tone is dry.
            - It lacks the specific "Chat Doctor" warmth (e.g., missing "Hi", "Thanks for your query", or "Regards").
            - It sounds like a textbook rather than a human doctor replying on a forum.

            **SCORE 7-10 (The "Chat Doctor" Success):**
            - The response sounds EXACTLY like the Reference Answer style.
            - It uses specific platform mannerisms (e.g., "Hi", "Welcome to Chat Doctor", "Hope I have answered your query", "Regards").
            - It feels like a human doctor typing a quick email/forum reply (even if it has minor grammar quirks, that's good! It shows it learned the dataset).
            - It shows empathy ("I understand your concern").

            **Note**: Do not judge based on medical accuracy unless it is dangerous. Judge based on whether the model **sounds like the specific doctor in the reference**.

            Provide a score from 1 to 10.
            Output format: JSON with keys 'score' and 'reason'.
            """
            
            try:
                response = self.client.chat.completions.create(
                    model=AZURE_DEPLOYMENT,
                    messages=[
                        {"role": "system", "content": "You are a helpful medical evaluator. Output strictly valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    response_format={ "type": "json_object" }
                )
                content = response.choices[0].message.content
                result = json.loads(content)
                scores.append(result)
            except Exception as e:
                print(f"[GPT-EVAL ERROR] {e}")
                scores.append({"score": 0, "reason": "Error"})
        return scores

def parse_args():
    parser = argparse.ArgumentParser(description="Full HealthCareMagic Evaluation Script")
    parser.add_argument("--model_path", type=str, default=None, help="Path to the saved LoRA checkpoint. If None, runs base model.")
    parser.add_argument("--base_model_name", type=str, default=None, help="Path to base model")
    parser.add_argument("--data_file", type=str, default=None, help="Path to local data or HF path")
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--output_file", type=str, default="eval_original_task/healthcaremagic/baseline.json")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limit samples for quick test")
    parser.add_argument("--use_gpt_eval", action="store_true", help="Enable GPT-4 evaluation (Costly!)")
    parser.add_argument("--gpt_sample_rate", type=float, default=0.2, help="Fraction of samples to send to GPT-4 (e.g. 0.1 for 10%)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Model Loading
    print(f"[INFO] Loading Tokenizer...")
    if args.model_path:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        except:
            tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_name, use_fast=True)
    
    tokenizer.padding_side = "left" # Critical for generation
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Loading Model on {device}...")
    
    model = ModulatedLlamaForCausalLM.from_pretrained(
        args.base_model_name,
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
        device_map=None
    )
    # Resize embeddings if needed
    if args.model_path:
        print(f"[INFO] Applying LoRA adapter: {args.model_path}")
        model = PeftModel.from_pretrained(model, args.model_path)
    else:
        print(f"[INFO] Running with Base Model Only (No LoRA)")

    model.to(device)
    model.eval()

    # 2. Data Loading
    print(f"[INFO] Loading HealthCareMagic Data...")
    # Mock config to re-use your existing Adapter logic if possible, or instantiate directly
    mock_config = {'data': {'type': 'healthcaremagic', 'healthcaremagic': {'eval_file': args.data_file}}}
    
    # Assuming you implemented HealthCareMagicAdapter as discussed previously
    adapter = HealthCareMagicAdapter(mock_config) 
    raw_data = adapter.load_data('test') # Ensure your adapter handles 'test' split logic
    
    if not raw_data:
        print("[ERROR] No data found.")
        return

    # Filter data? HealthCare inputs can be long.
    print("[INFO] Pre-processing data...")
    temp_dataset = UnifiedChatDataset(raw_data, tokenizer, max_length=1024)
    eval_samples = temp_dataset.raw_samples
    
    if args.limit:
        eval_samples = eval_samples[:args.limit]
    
    inference_dataset = InferenceDataset(eval_samples, tokenizer)
    collator = DataCollatorForGeneration(tokenizer)
    dataloader = DataLoader(inference_dataset, batch_size=args.batch_size, collate_fn=collator, shuffle=False)

    # 3. Generation Loop
    print(f"[INFO] Generating Responses for {len(eval_samples)} samples...")
    predictions = []
    references = []
    prompts_list = []
    
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]

    for batch in tqdm(dataloader, desc="Generating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=256,      # Medical answers need more space than emails
                min_new_tokens=10,
                repetition_penalty=1.1, 
                do_sample=True,          # Sampling usually better for dialogue
                temperature=0.6,         # Slight creativity, but mostly deterministic
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=terminators
            )
        
        input_len = input_ids.shape[1]
        decoded = tokenizer.batch_decode(outputs[:, input_len:], skip_special_tokens=True)
        
        predictions.extend([p.strip() for p in decoded])
        references.extend(batch['references'])
        prompts_list.extend(batch['prompts'])

    # 4. Traditional Metrics Evaluation
    print("[INFO] Computing ROUGE & BERTScore...")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    
    rouge_res = rouge.compute(predictions=predictions, references=references)
    bert_res = bertscore.compute(predictions=predictions, references=references, lang="en")
    bert_f1 = np.mean(bert_res['f1'])
    
    print(f"ROUGE-L: {rouge_res['rougeL']:.4f} | BERTScore F1: {bert_f1:.4f}")

    # 5. LLM-as-a-Judge Evaluation (Optional but recommended)
    gpt_scores = []
    if args.use_gpt_eval:
        print("[INFO] Starting GPT-4 Evaluation (Azure OpenAI)...")
        evaluator = GPTEvaluator()
        
        # Subsample for GPT eval to save time if needed
        indices = np.random.choice(len(predictions), int(len(predictions) * args.gpt_sample_rate), replace=False)
        sub_inputs = [prompts_list[i] for i in indices]
        sub_refs = [references[i] for i in indices]
        sub_preds = [predictions[i] for i in indices]
        
        # Process strictly sequentially or use asyncio for speed (Sequential is safer for rate limits)
        print(f"[INFO] Sending {len(sub_inputs)} samples to GPT-4...")
        gpt_results = evaluator.evaluate_batch(sub_inputs, sub_refs, sub_preds)
        
        # Calculate average GPT score
        valid_scores = [r['score'] for r in gpt_results if isinstance(r['score'], (int, float))]
        avg_gpt_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0
        print(f"GPT-4 Average Score: {avg_gpt_score:.2f} / 10")
        
        # Merge results for saving
        for idx, res in zip(indices, gpt_results):
            gpt_scores.append({
                "idx": int(idx),
                "gpt_score": res['score'],
                "gpt_reason": res['reason']
            })

    # 6. Save Everything
    final_log = []
    gpt_map = {item['idx']: item for item in gpt_scores}
    
    for i in range(len(predictions)):
        entry = {
            "input": prompts_list[i],
            "target": references[i],
            "prediction": predictions[i]
        }
        if i in gpt_map:
            entry["gpt_eval"] = gpt_map[i]
        final_log.append(entry)

    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(final_log, f, indent=2, ensure_ascii=False)
        
    metrics = {
        "rouge": rouge_res,
        "bertscore": bert_f1,
        "gpt4_score": avg_gpt_score if args.use_gpt_eval else None
    }
    with open(args.output_file.replace('.json', '_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print("[INFO] Evaluation Complete.")

if __name__ == "__main__":
    main()