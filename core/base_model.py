import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, DPOTrainer, SFTConfig
from transformers import TrainingArguments
from datasets import Dataset
from typing import List, Dict
from pathlib import Path
from rouge_score import rouge_scorer

# Try to import DPOConfig for proper DPO configuration
try:
    from trl import DPOConfig

    DPO_CONFIG_AVAILABLE = True
except ImportError:
    DPO_CONFIG_AVAILABLE = False
    print(
        "⚠️ DPOConfig not available in this TRL version - using TrainingArguments fallback"
    )

from utils.logger import CompetitionLogger
from core.ml_model import ClinicalExample


class BaseUnslothModel:
    _model_cache = {}
    _tokenizer_cache = {}

    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config["model"]
        self.training_config = config["training"]
        self.model_name = self.model_config["name"]

        model_output_dir = self.config.get("model_output_dir", "models")
        self.sft_model_path = (
            Path(model_output_dir) / f"{self.model_name.replace('/', '_')}_sft"
        )
        self.dpo_model_path = (
            Path(model_output_dir) / f"{self.model_name.replace('/', '_')}_dpo"
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = CompetitionLogger(self.__class__.__name__)

        self.model, self.tokenizer = self._load_model()
        self._configure_for_finetuning()
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def _load_model(self):
        cache_key = f"{self.model_name}_{self.model_config['load_in_4bit']}"
        if cache_key in self._model_cache:
            self.logger.info("✅ Using cached model from memory")
            return self._model_cache[cache_key], self._tokenizer_cache[cache_key]

        self.logger.info(f"Downloading/Loading from cache: {self.model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.model_config["max_seq_length"],
            dtype=None,
            load_in_4bit=self.model_config["load_in_4bit"],
            cache_dir=self.model_config["cache_dir"],
        )

        self._model_cache[cache_key] = model
        self._tokenizer_cache[cache_key] = tokenizer
        self.logger.info("✅ Model cached in memory for future use")
        return model, tokenizer

    def _configure_for_finetuning(self):
        lora_config = self.training_config["lora"]
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config["r"],
            target_modules=lora_config["target_modules"],
            lora_alpha=lora_config["lora_alpha"],
            lora_dropout=lora_config["lora_dropout"],
            bias=lora_config["bias"],
            use_gradient_checkpointing=lora_config["use_gradient_checkpointing"],
            random_state=lora_config["random_state"],
            use_rslora=lora_config["use_rslora"],
            loftq_config=lora_config.get("loftq_config"),
        )

    def fine_tune(
        self,
        train_examples: List[ClinicalExample],
        val_examples: List[ClinicalExample] = None,
    ):
        train_texts = [example.input_text for example in train_examples]
        train_dataset = Dataset.from_dict({"text": train_texts})

        sft_args = self.training_config["sft_config"]
        sft_args["max_steps"] = (
            len(train_examples)
            // sft_args["per_device_train_batch_size"]
            * self.training_config["epochs"]
        )
        sft_args["fp16"] = not torch.cuda.is_bf16_supported()
        sft_args["bf16"] = torch.cuda.is_bf16_supported()

        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            args=SFTConfig(**sft_args),
        )

        self.logger.info(f"Starting fine-tuning for {self.model_name}...")
        trainer.train()

        results = {"training_stats": trainer.state.log_history}
        if val_examples:
            rouge_scores = self._evaluate_rouge(val_examples[:10])
            results["validation_rouge"] = rouge_scores
            self.logger.info(f"Validation ROUGE-L: {rouge_scores['rougeL']:.4f}")
        return results    
    def dpo_fine_tune(self, dpo_dataset: Dataset):
        """Fine-tune the model using Direct Preference Optimization (DPO) with enhanced compatibility."""

        self.logger.info(f"Starting DPO fine-tuning for {self.model_name}...")

        # Use modern compatibility approach - safe parameter extraction
        dpo_config = {
            "output_dir": str(self.dpo_model_path),
            "num_train_epochs": self.config.get("dpo_epochs", 1),
            "per_device_train_batch_size": self.config.get("dpo_batch_size", 1),
            "gradient_accumulation_steps": self.config.get("dpo_gradient_accumulation_steps", 4),
            "learning_rate": self.config.get("dpo_learning_rate", 1e-7),
            "logging_steps": self.config.get("dpo_logging_steps", 10),
            "save_steps": self.config.get("dpo_save_steps", 100),
            "save_total_limit": self.config.get("dpo_save_total_limit", 2),
            "beta": self.config.get("dpo_beta", 0.1),
            "max_length": self.config.get("dpo_max_seq_length", 1024),
            "max_prompt_length": self.config.get("dpo_max_prompt_length", 512),
        }

        try:
            # Import the compatibility fix if available (from notebook)
            try:
                from __main__ import DPOCompatibilityFix
                self.logger.info("Using DPO compatibility fix from notebook")
                
                dpo_trainer = DPOCompatibilityFix.create_dpo_trainer_safe(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    train_dataset=dpo_dataset,
                    **dpo_config
                )
                
            except ImportError:
                self.logger.info("Compatibility fix not available, using standard approach")
                
                # Fallback to standard TrainingArguments with safe parameters
                safe_params = {
                    "output_dir": dpo_config["output_dir"],
                    "num_train_epochs": dpo_config["num_train_epochs"],
                    "per_device_train_batch_size": dpo_config["per_device_train_batch_size"],
                    "gradient_accumulation_steps": dpo_config["gradient_accumulation_steps"],
                    "learning_rate": dpo_config["learning_rate"],
                    "logging_steps": dpo_config["logging_steps"],
                    "save_steps": dpo_config["save_steps"],
                    "save_total_limit": dpo_config["save_total_limit"],
                    "lr_scheduler_type": "cosine",
                    "optim": "adamw_torch",
                    "warmup_ratio": 0.1,
                    "report_to": "none",
                    "remove_unused_columns": False,
                }
                
                dpo_args = TrainingArguments(**safe_params)
                
                # Add minimal compatibility attributes
                essential_attrs = {
                    "padding_value": getattr(self.tokenizer, 'pad_token_id', -100),
                    "reference_free": True,
                    "disable_dropout": True,
                }
                
                for attr_name, attr_value in essential_attrs.items():
                    if not hasattr(dpo_args, attr_name):
                        setattr(dpo_args, attr_name, attr_value)

                # Initialize DPOTrainer with minimal parameters
                dpo_trainer = DPOTrainer(
                    model=self.model,
                    ref_model=None,  # Reference-free approach
                    args=dpo_args,
                    beta=dpo_config["beta"],
                    train_dataset=dpo_dataset,
                    tokenizer=self.tokenizer,
                    max_length=dpo_config["max_length"],
                    max_prompt_length=dpo_config["max_prompt_length"],
                )

            # Train the model
            self.logger.info("Starting DPO training...")
            dpo_trainer.train()

            self.logger.info(f"DPO fine-tuning completed for {self.model_name}.")
            return {"dpo_training_stats": dpo_trainer.state.log_history}

        except Exception as e:
            self.logger.error(f"DPO training failed with error: {e}")
            
            # Enhanced error reporting
            error_type = type(e).__name__
            error_message = str(e)
            
            if "liger" in error_message.lower():
                self.logger.info("Liger kernel related error - this is a known compatibility issue")
            elif "attribute" in error_message.lower() and "TrainingArguments" in error_message:
                self.logger.info("TrainingArguments compatibility error - version mismatch detected")
            
            # Return error details for debugging
            return {
                "error": error_message,
                "error_type": error_type,
                "status": "failed",
                "fallback_available": "sft_model_ready"
            }

    def generate_response(self, input_prompt: str, max_length: int = 512) -> str:
        raise NotImplementedError("Subclasses must implement generate_response")

    def _evaluate_rouge(self, examples: List[ClinicalExample]) -> Dict[str, float]:
        predictions = []
        references = []

        for example in examples:
            prompt = self._extract_prompt_from_chat_template(example.input_text)
            pred = self.generate_response(prompt, max_length=400)
            predictions.append(pred)
            references.append(example.target_response)

        rouge_scores = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores["rouge1"] += scores["rouge1"].fmeasure
            rouge_scores["rouge2"] += scores["rouge2"].fmeasure
            rouge_scores["rougeL"] += scores["rougeL"].fmeasure

        n = len(predictions)
        return {k: v / n for k, v in rouge_scores.items()}

    def _extract_prompt_from_chat_template(self, text: str) -> str:
        raise NotImplementedError(
            "Subclasses must implement _extract_prompt_from_chat_template"
        )

    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.logger.info(f"Model saved to {path}")

    def cleanup_model(self):
        if hasattr(self, "model") and self.model is not None:
            del self.model
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            del self.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info(f"{self.model_name} cleaned up from memory")

    @classmethod
    def clear_cache(cls):
        cls._model_cache.clear()
        cls._tokenizer_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        CompetitionLogger("BaseUnslothModel").info("All model caches cleared")
