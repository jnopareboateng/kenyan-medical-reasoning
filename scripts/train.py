"""
Main Training Script for Clinical Reasoning Models

This script orchestrates the fine-tuning and evaluation of models based on a given configuration file.
"""

import argparse
import pandas as pd
from utils.logger import CompetitionLogger
from utils.paths import get_project_paths, load_config
from core.qwen3_model import ClinicalQwen3Model
from core.llama32_model import ClinicalLlama32Model

# Model mapping
MODEL_MAPPING = {
    "Qwen3": ClinicalQwen3Model,
    "Llama32": ClinicalLlama32Model,
}

def main(config_path: str):
    """Main function to run the training pipeline."""
    
    # 1. Setup
    logger = CompetitionLogger("MainTrainingScript")
    logger.info(f"Starting training pipeline with config: {config_path}")
    paths = get_project_paths()
    config = load_config(config_path)

    # 2. Select and initialize model
    model_provider = config.get('model', {}).get('provider', '')
    if model_provider not in MODEL_MAPPING:
        raise ValueError(f"Unsupported model provider: {model_provider}")
    
    ModelClass = MODEL_MAPPING[model_provider]
    model = ModelClass(config)
    logger.info(f"Initialized model: {model.model_name}")

    # 3. Load and prepare data
    train_df = pd.read_csv(paths['train_data'])
    training_examples = model.prepare_training_data(train_df)
    logger.info(f"Prepared {len(training_examples)} training examples.")

    # 4. Split data
    train_size = int(0.85 * len(training_examples))
    train_examples = training_examples[:train_size]
    val_examples = training_examples[train_size:]
    logger.info(f"Training set size: {len(train_examples)}, Validation set size: {len(val_examples)}")

    # 5. Fine-tune the model
    training_results = model.fine_tune(
        train_examples=train_examples, 
        val_examples=val_examples
    )
    logger.info(f"Fine-tuning complete. Results: {training_results}")

    # 6. Generate predictions on test set
    test_df = pd.read_csv(paths['test_data'])
    predictions = []
    for _, row in test_df.iterrows():
        input_prompt = model._create_input_prompt(row)
        response = model.generate_response(input_prompt)
        predictions.append(response)
    logger.info(f"Generated {len(predictions)} predictions on the test set.")

    # 7. Save submission file
    submission_df = pd.DataFrame({
        "Master_Index": test_df["Master_Index"],
        "Clinician": predictions,
    })
    submission_path = paths['results'] / f"{model.model_name.replace('/', '_')}_submission.csv"
    submission_df.to_csv(submission_path, index=False)
    logger.info(f"Submission file saved to: {submission_path}")

    # 8. Save the model
    model_save_path = paths['models'] / f"{model.model_name.replace('/', '_')}_finetuned"
    model.save_model(str(model_save_path))

    logger.info("Training pipeline finished successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the clinical model training pipeline.")
    parser.add_argument("--config", type=str, required=True, help="Path to the model configuration YAML file.")
    args = parser.parse_args()
    
    main(args.config)
