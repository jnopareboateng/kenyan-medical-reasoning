import torch
from unsloth import FastLanguageModel
from trl import SFTTrainer, DPOTrainer, SFTConfig
from transformers import TrainingArguments
from datasets import Dataset
from typing import List, Dict
from pathlib import Path
from rouge_score import rouge_scorer

from utils.logger import CompetitionLogger
from core.ml_model import ClinicalExample

class BaseUnslothModel:
    _model_cache = {}
    _tokenizer_cache = {}

    def __init__(self, config: Dict):
        self.config = config
        self.model_config = config['model']
        self.training_config = config['training']
        self.model_name = self.model_config['name']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = CompetitionLogger(self.__class__.__name__)
        
        self.model, self.tokenizer = self._load_model()
        self._configure_for_finetuning()
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def _load_model(self):
        cache_key = f"{self.model_name}_{self.model_config['load_in_4bit']}"
        if cache_key in self._model_cache:
            self.logger.info("✅ Using cached model from memory")
            return self._model_cache[cache_key], self._tokenizer_cache[cache_key]

        self.logger.info(f"Downloading/Loading from cache: {self.model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.model_config['max_seq_length'],
            dtype=None,
            load_in_4bit=self.model_config['load_in_4bit'],
            cache_dir=self.model_config['cache_dir'],
        )
        
        self._model_cache[cache_key] = model
        self._tokenizer_cache[cache_key] = tokenizer
        self.logger.info("✅ Model cached in memory for future use")
        return model, tokenizer

    def _configure_for_finetuning(self):
        lora_config = self.training_config['lora']
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=lora_config['r'],
            target_modules=lora_config['target_modules'],
            lora_alpha=lora_config['lora_alpha'],
            lora_dropout=lora_config['lora_dropout'],
            bias=lora_config['bias'],
            use_gradient_checkpointing=lora_config['use_gradient_checkpointing'],
            random_state=lora_config['random_state'],
            use_rslora=lora_config['use_rslora'],
            loftq_config=lora_config.get('loftq_config')
        )

    def fine_tune(self, train_examples: List[ClinicalExample], val_examples: List[ClinicalExample] = None):
        train_texts = [example.input_text for example in train_examples]
        train_dataset = Dataset.from_dict({"text": train_texts})
        
        sft_args = self.training_config['sft_config']
        sft_args['max_steps'] = len(train_examples) // sft_args['per_device_train_batch_size'] * self.training_config['epochs']
        sft_args['fp16'] = not torch.cuda.is_bf16_supported()
        sft_args['bf16'] = torch.cuda.is_bf16_supported()

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
        
        dpo_config = self.config['dpo_training']
        self.logger.info(f"Starting DPO fine-tuning for {self.model_name}...")

        # Use DPOConfig instead of TrainingArguments for unsloth compatibility
        try:
            from trl import DPOConfig
            
            training_args = DPOConfig(
                per_device_train_batch_size=dpo_config.get('batch_size', 1),
                gradient_accumulation_steps=dpo_config.get('gradient_accumulation_steps', 8),
                warmup_steps=dpo_config.get('warmup_steps', 5),
                learning_rate=dpo_config.get('learning_rate', 5e-7),
                num_train_epochs=dpo_config.get('epochs', 2),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs/dpo",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                report_to=None,  # Disable wandb/tensorboard
                save_strategy="no",  # Don't save checkpoints during DPO
                beta=dpo_config.get('beta', 0.1),  # Beta parameter for DPO
                max_length=1024,
                max_prompt_length=512,
                padding_value=self.tokenizer.pad_token_id if hasattr(self.tokenizer, 'pad_token_id') else 0,
            )
            
        except ImportError:
            # Fallback to TrainingArguments if DPOConfig is not available
            self.logger.warning("DPOConfig not available, using TrainingArguments")
            training_args = TrainingArguments(
                per_device_train_batch_size=dpo_config.get('batch_size', 1),
                gradient_accumulation_steps=dpo_config.get('gradient_accumulation_steps', 8),
                warmup_steps=dpo_config.get('warmup_steps', 5),
                learning_rate=dpo_config.get('learning_rate', 5e-7),
                num_train_epochs=dpo_config.get('epochs', 2),
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs/dpo",
                remove_unused_columns=False,
                dataloader_pin_memory=False,
                report_to=None,
                save_strategy="no",
            )

        try:
            # Initialize the DPOTrainer with minimal parameters for unsloth compatibility
            dpo_trainer_kwargs = {
                "model": self.model,
                "args": training_args,
                "train_dataset": dpo_dataset,
                "processing_class": self.tokenizer,  # Use processing_class instead of tokenizer
            }
            
            # Only add DPO-specific parameters if NOT using DPOConfig
            if 'DPOConfig' not in str(type(training_args)):
                dpo_trainer_kwargs["beta"] = dpo_config.get('beta', 0.1)
                dpo_trainer_kwargs["max_length"] = 1024
                dpo_trainer_kwargs["max_prompt_length"] = 512
            
            dpo_trainer = DPOTrainer(**dpo_trainer_kwargs)

            # Train the model
            dpo_trainer.train()

            self.logger.info(f"DPO fine-tuning completed for {self.model_name}.")
            return {"dpo_training_stats": dpo_trainer.state.log_history}
            
        except Exception as e:
            self.logger.error(f"DPO training failed with error: {e}")
            
            # Fallback: Try with even more minimal configuration
            self.logger.info("Attempting DPO training with ultra-minimal configuration...")
            
            try:
                # Most minimal DPO trainer possible
                simple_args = TrainingArguments(
                    per_device_train_batch_size=1,
                    gradient_accumulation_steps=4,
                    num_train_epochs=1,
                    learning_rate=1e-7,
                    logging_steps=5,
                    output_dir="outputs/dpo_minimal",
                    report_to=None,
                    save_strategy="no",
                    remove_unused_columns=False,
                )
                
                # Try with absolute minimal parameters
                dpo_trainer_simple = DPOTrainer(
                    model=self.model,
                    args=simple_args,
                    train_dataset=dpo_dataset,
                    processing_class=self.tokenizer,
                )
                
                dpo_trainer_simple.train()
                self.logger.info("DPO training completed with ultra-minimal configuration.")
                return {"dpo_training_stats": "minimal_completed"}
                
            except Exception as e2:
                self.logger.error(f"Fallback DPO training also failed: {e2}")
                # Instead of raising, return None to allow pipeline to continue
                self.logger.warning("DPO training failed completely. Continuing with SFT model only.")
                return None

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
        
        rouge_scores = {'rouge1': 0.0, 'rouge2': 0.0, 'rougeL': 0.0}
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'] += scores['rouge1'].fmeasure
            rouge_scores['rouge2'] += scores['rouge2'].fmeasure
            rouge_scores['rougeL'] += scores['rougeL'].fmeasure
        
        n = len(predictions)
        return {k: v/n for k, v in rouge_scores.items()}

    def _extract_prompt_from_chat_template(self, text: str) -> str:
        raise NotImplementedError("Subclasses must implement _extract_prompt_from_chat_template")

    def save_model(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        self.logger.info(f"Model saved to {path}")

    def cleanup_model(self):
        if hasattr(self, 'model') and self.model is not None:
            del self.model
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
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

