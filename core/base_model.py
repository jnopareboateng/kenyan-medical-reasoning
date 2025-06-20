
import torch
from unsloth import FastLanguageModel
from transformers import SFTConfig
from trl import SFTTrainer
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

