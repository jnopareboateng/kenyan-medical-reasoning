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
        """Fine-tune the model using Direct Preference Optimization (DPO)."""

        self.logger.info(f"Starting DPO fine-tuning for {self.model_name}...")

        dpo_epochs = self.config.get("dpo_epochs", 1)
        dpo_learning_rate = self.config.get("dpo_learning_rate", 5e-7)
        dpo_beta = self.config.get("dpo_beta", 0.1)

        try:
            # First attempt: Try with DPOConfig if available
            try:
                from trl import DPOConfig
                
                self.logger.info("Using DPOConfig for training configuration...")
                
                # Use DPOConfig which properly supports DPO-specific parameters
                dpo_config = DPOConfig(
                    output_dir=str(self.dpo_model_path),
                    num_train_epochs=dpo_epochs,
                    per_device_train_batch_size=self.config.get("dpo_batch_size", 1),
                    gradient_accumulation_steps=self.config.get("dpo_gradient_accumulation_steps", 4),
                    learning_rate=dpo_learning_rate,
                    logging_steps=self.config.get("dpo_logging_steps", 10),
                    save_steps=self.config.get("dpo_save_steps", 100),
                    save_total_limit=self.config.get("dpo_save_total_limit", 2),
                    lr_scheduler_type="cosine",
                    optim="adamw_torch",
                    warmup_ratio=0.1,
                    report_to="none",
                    
                    # DPO-specific parameters (these belong in DPOConfig)
                    beta=dpo_beta,
                    max_length=self.config.get("dpo_max_seq_length", 1024),
                    max_prompt_length=self.config.get("dpo_max_prompt_length", 512),
                    loss_type="sigmoid",
                    label_smoothing=0.0,
                    reference_free=True,
                )

                # Initialize DPOTrainer with DPOConfig
                dpo_trainer = DPOTrainer(
                    model=self.model,
                    ref_model=None,
                    args=dpo_config,
                    train_dataset=dpo_dataset,
                    tokenizer=self.tokenizer,
                )

                dpo_trainer.train()
                self.logger.info(f"DPO fine-tuning completed using DPOConfig for {self.model_name}.")
                return {"dpo_training_stats": dpo_trainer.state.log_history}

            except ImportError:
                self.logger.info("DPOConfig not available, using TrainingArguments approach...")
                raise Exception("Fallback to TrainingArguments")

        except Exception:
            # Second attempt: TrainingArguments without problematic parameters
            self.logger.info("Using TrainingArguments with DPO parameters passed to trainer...")
            
            try:
                # Clean TrainingArguments - NO DPO-specific parameters here
                training_args = TrainingArguments(
                    output_dir=str(self.dpo_model_path),
                    num_train_epochs=dpo_epochs,
                    per_device_train_batch_size=self.config.get("dpo_batch_size", 1),
                    gradient_accumulation_steps=self.config.get("dpo_gradient_accumulation_steps", 4),
                    learning_rate=dpo_learning_rate,
                    logging_steps=self.config.get("dpo_logging_steps", 10),
                    save_steps=self.config.get("dpo_save_steps", 100),
                    save_total_limit=self.config.get("dpo_save_total_limit", 2),
                    lr_scheduler_type="cosine",
                    optim="adamw_torch",
                    warmup_ratio=0.1,
                    report_to="none",
                    remove_unused_columns=False,
                    dataloader_drop_last=False,
                )

                # Initialize DPOTrainer with parameters passed directly (not in args)
                dpo_trainer = DPOTrainer(
                    model=self.model,
                    ref_model=None,  # Reference-free DPO
                    args=training_args,
                    beta=dpo_beta,
                    train_dataset=dpo_dataset,
                    tokenizer=self.tokenizer,
                    max_length=self.config.get("dpo_max_seq_length", 1024),
                    max_prompt_length=self.config.get("dpo_max_prompt_length", 512),
                    loss_type="sigmoid",
                    label_smoothing=0.0,
                    # Note: disable_dropout and other model-specific params removed
                    # as they may not be supported in all TRL versions
                )

                dpo_trainer.train()
                self.logger.info(f"DPO fine-tuning completed using TrainingArguments for {self.model_name}.")
                return {"dpo_training_stats": dpo_trainer.state.log_history}

            except Exception as e:
                self.logger.error(f"Standard DPO training failed: {e}")
                
                # Third attempt: Ultra-minimal configuration
                self.logger.info("Attempting ultra-minimal DPO configuration...")
                
                try:
                    # Absolutely minimal TrainingArguments
                    minimal_args = TrainingArguments(
                        output_dir=f"{str(self.dpo_model_path)}_minimal",
                        num_train_epochs=1,
                        per_device_train_batch_size=1,
                        gradient_accumulation_steps=1,
                        learning_rate=1e-7,
                        logging_steps=50,
                        save_steps=1000,
                        report_to="none",
                    )

                    # Absolutely minimal DPOTrainer
                    dpo_trainer_minimal = DPOTrainer(
                        model=self.model,
                        ref_model=None,
                        args=minimal_args,
                        beta=0.1,
                        train_dataset=dpo_dataset,
                        tokenizer=self.tokenizer,
                        max_length=512,
                        max_prompt_length=256,
                    )
                    
                    dpo_trainer_minimal.train()
                    self.logger.info("Ultra-minimal DPO training completed successfully.")
                    self.model = dpo_trainer_minimal.model
                    return {"dpo_training_stats": "minimal_dpo_completed"}

                except Exception as e2:
                    self.logger.error(f"Ultra-minimal DPO training failed: {e2}")
                    
                    # Fourth attempt: Basic supervised fine-tuning on chosen responses
                    self.logger.info("Falling back to supervised fine-tuning on chosen responses...")
                    
                    try:
                        from transformers import Trainer
                        
                        # Simple supervised fine-tuning arguments
                        sft_args = TrainingArguments(
                            output_dir=f"{str(self.dpo_model_path)}_sft_fallback",
                            num_train_epochs=1,
                            per_device_train_batch_size=1,
                            gradient_accumulation_steps=2,
                            learning_rate=1e-6,
                            logging_steps=100,
                            save_steps=1000,
                            report_to="none",
                            remove_unused_columns=False,
                        )
                        
                        # Convert DPO dataset to supervised format using chosen responses
                        def format_for_supervised(example):
                            # Combine prompt and chosen response for supervised training
                            full_text = example["prompt"] + example["chosen"]
                            
                            # Tokenize
                            tokenized = self.tokenizer(
                                full_text,
                                truncation=True,
                                max_length=512,
                                padding="max_length",
                                return_tensors="pt"
                            )
                            
                            return {
                                "input_ids": tokenized["input_ids"].squeeze(),
                                "attention_mask": tokenized["attention_mask"].squeeze(),
                                "labels": tokenized["input_ids"].squeeze(),  # Same as input_ids for causal LM
                            }
                        
                        # Apply the transformation
                        sft_dataset = dpo_dataset.map(format_for_supervised, remove_columns=dpo_dataset.column_names)
                        
                        # Basic trainer for supervised fine-tuning
                        sft_trainer = Trainer(
                            model=self.model,
                            args=sft_args,
                            train_dataset=sft_dataset,
                            tokenizer=self.tokenizer,
                        )
                        
                        sft_trainer.train()
                        self.logger.info("Supervised fine-tuning fallback completed successfully.")
                        self.model = sft_trainer.model
                        return {"dpo_training_stats": "sft_fallback_completed"}
                        
                    except Exception as e3:
                        self.logger.error(f"All training approaches failed: {e3}")
                        raise Exception(f"Complete training failure. Last error: {e3}")

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
