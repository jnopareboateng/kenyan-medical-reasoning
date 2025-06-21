"""
Main DPO Training Script for Clinical Reasoning Models

This script orchestrates the DPO fine-tuning of a pre-trained SFT model.
"""

import argparse
import pandas as pd
from datasets import load_dataset

from utils.logger import CompetitionLogger
from utils.paths import get_project_paths, load_config
from core.qwen3_model import ClinicalQwen3Model
from core.llama32_model import ClinicalLlama32Model
from core.gemma2_model import ClinicalGemma2Model

# Model mapping
MODEL_MAPPING = {
    "Qwen3": ClinicalQwen3Model,
    "Llama32": ClinicalLlama32Model,
    "google": ClinicalGemma2Model,
}

def main(config_path: str):
    """Main function to run the DPO training pipeline."""
    
    logger = CompetitionLogger("DPO_TrainingScript")
    logger.info(f"Starting DPO training pipeline with config: {config_path}")
    paths = get_project_paths()
    config = load_config(config_path)

    # 1. Select and initialize model
    model_provider = config.get('model', {}).get('provider', '')
    if model_provider not in MODEL_MAPPING:
        raise ValueError(f"Unsupported model provider: {model_provider}")
    
    ModelClass = MODEL_MAPPING[model_provider]
    model = ModelClass(config)
    logger.info(f"Initialized base model: {model.model_name}")    # 2. Load the SFT-trained model 
    sft_model_path = config.get('dpo_training', {}).get('sft_model_path')
    if not sft_model_path:
        raise ValueError("sft_model_path not specified in config for DPO training")
    
    # Initialize model and load SFT weights
    logger.info(f"Loading SFT model from: {sft_model_path}")
    # Note: The model loading with adapter will be handled in the model class

    # 3. Load DPO dataset
    dpo_dataset_path = paths['data'] / "dpo_train_dataset.jsonl"
    if not dpo_dataset_path.exists():
        raise FileNotFoundError(f"DPO dataset not found. Please run scripts/prepare_dpo_data.py first.")
    
    dpo_dataset = load_dataset("json", data_files=str(dpo_dataset_path), split="train")
    logger.info(f"Loaded DPO dataset with {len(dpo_dataset)} examples.")

    # 4. Run DPO fine-tuning
    dpo_results = model.dpo_fine_tune(dpo_dataset)
    logger.info(f"DPO fine-tuning complete. Results: {dpo_results}")

    # 5. Save the DPO-tuned model
    dpo_model_save_path = paths['models'] / f"{model.model_name.replace('/', '_')}_dpo_finetuned"
    model.save_model(str(dpo_model_save_path))

    logger.info("DPO Training pipeline finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the clinical model DPO training pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration YAML file.")
    args = parser.parse_args()
    
    main(args.config)
