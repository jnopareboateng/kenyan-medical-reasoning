"""
Prepare Dataset for Direct Preference Optimization (DPO)

This script processes the training data to create a dataset suitable for DPO.
The dataset consists of triplets: (prompt, chosen_response, rejected_response).
"""

import pandas as pd
import json
import argparse
import os
import sys
from tqdm import tqdm
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.paths import get_project_paths, load_config
from utils.logger import CompetitionLogger
from core.qwen3_model import ClinicalQwen3Model # Using Qwen3 for prompt creation logic

def create_dpo_dataset(config_path: str):
    """Generate the DPO dataset from the raw training data."""
    
    logger = CompetitionLogger("DPO_DataPrep")
    paths = get_project_paths()
    config = load_config(config_path)
    
    # Use a model class to get the prompt creation logic
    # This ensures prompts are consistent with SFT training
    model_for_prompting = ClinicalQwen3Model(config)
    
    train_df = pd.read_csv(paths['train_data'])
    logger.info(f"Loaded {len(train_df)} cases from train.csv")
    
    dpo_examples = []
    ai_response_cols = ['GPT4.0', 'LLAMA', 'GEMINI']
    
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Processing cases"):
        # Ensure there is a valid expert response
        if pd.isna(row['Clinician']) or len(str(row['Clinician']).strip()) < 50:
            continue
            
        # Find the best rejected response
        available_rejected = []
        for col in ai_response_cols:
            if pd.notna(row.get(col)) and len(str(row.get(col)).strip()) > 50:
                available_rejected.append(str(row[col]))
        
        if not available_rejected:
            continue # Skip if no valid AI responses to use as rejected
            
        # Heuristic: shortest response is likely the least comprehensive
        rejected_response = min(available_rejected, key=len)
        chosen_response = str(row['Clinician']).strip()
        
        # Ensure chosen and rejected are not identical
        if rejected_response == chosen_response:
            continue

        # Create the prompt
        prompt = model_for_prompting._create_input_prompt(row)
        
        dpo_examples.append({
            "prompt": prompt,
            "chosen": chosen_response,
            "rejected": rejected_response
        })
        
    logger.info(f"Created {len(dpo_examples)} valid DPO examples.")
    
    # Save to JSON Lines file
    output_path = paths['data'] / "dpo_train_dataset.jsonl"
    with open(output_path, 'w', encoding='utf-8') as f:
        for example in dpo_examples:
            f.write(json.dumps(example) + "\n")
            
    logger.info(f"DPO dataset saved to {output_path}")
    print(f"✅ DPO dataset created at: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DPO dataset.")
    parser.add_argument("--config", type=str, default="configs/qwen3.yaml", help="Path to a model configuration file to use for prompt generation logic.")
    args = parser.parse_args()
    
    create_dpo_dataset(args.config)
