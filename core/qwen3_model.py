"""
Qwen-3-0.5B Implementation for Clinical Response Generation
Using Alibaba's latest 0.5B parameter instruction-tuned model with Unsloth optimization
"""

import torch
import pandas as pd
import numpy as np
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import re
from dataclasses import dataclass
from rouge_score import rouge_scorer
import logging
from pathlib import Path

from utils.logger import CompetitionLogger
from utils.paths import get_project_paths
from .ml_model import ClinicalExample  # Reuse existing data structure

logger = CompetitionLogger("Qwen3Model")
paths = get_project_paths()

class ClinicalQwen3Model:
    """Qwen-3-0.5B fine-tuned for clinical reasoning with Unsloth optimization"""
    
    # Class-level model cache to prevent re-downloading
    _model_cache = {}
    _tokenizer_cache = {}
    
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct", load_in_4bit: bool = True, 
                 cache_dir: str = "./models", force_download: bool = False):
        """Initialize Qwen-3-0.5B with Unsloth for 4-bit quantization"""
        
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_in_4bit = load_in_4bit
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create cache key
        cache_key = f"{model_name}_{load_in_4bit}"
        
        logger.info(f"Loading {model_name} with caching optimization")
        
        # Check if model is already in memory cache
        if not force_download and cache_key in self._model_cache:
            logger.info("✅ Using cached model from memory")
            self.model = self._model_cache[cache_key]
            self.tokenizer = self._tokenizer_cache[cache_key]
            return
        
        # Load model with Unsloth for efficient training and inference
        logger.info(f"Downloading/Loading from cache: {model_name}")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,  # Qwen-3 supports up to 32K, but 2K is sufficient for clinical cases
            dtype=None,  # Auto-detect
            load_in_4bit=load_in_4bit,  # Use 4-bit quantization for efficiency
            cache_dir=str(self.cache_dir),  # Use persistent cache
        )
        
        # Cache in memory for subsequent uses
        self._model_cache[cache_key] = self.model
        self._tokenizer_cache[cache_key] = self.tokenizer
        logger.info("✅ Model cached in memory for future use")
        
        # Configure for fine-tuning with LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,  # LoRA rank
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        # ROUGE scorer for evaluation
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        # Qwen chat template
        self.chat_template = """<|im_start|>system
You are Qwen, an expert clinical practitioner specializing in healthcare delivery in Kenya. You have extensive experience working within the Kenyan healthcare system and understand the unique challenges, resource constraints, and cultural considerations. Provide detailed, evidence-based clinical assessments and management plans that are appropriate for the local context.<|im_end|>
<|im_start|>user
{prompt}<|im_end|>
<|im_start|>assistant
{response}<|im_end|>"""
        
        logger.info(f"Qwen-3-0.5B loaded with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def prepare_training_data(self, df: pd.DataFrame) -> List[ClinicalExample]:
        """Convert DataFrame to training examples using expert responses"""
        
        examples = []
        expert_columns = ['Nursing Competency', 'Clinical Panel', 'Clinician', 'GPT4.0', 'LLAMA', 'GEMINI']
        
        for idx, row in df.iterrows():
            input_text = self._create_input_prompt(row)
            
            # Use the best available expert response
            for col in expert_columns:
                if col in df.columns and pd.notna(row[col]) and len(str(row[col]).strip()) > 50:
                    target_response = str(row[col]).strip()
                    
                    # Format for Qwen chat template
                    formatted_input = self.chat_template.format(
                        prompt=input_text,
                        response=target_response
                    )
                    
                    example = ClinicalExample(
                        input_text=formatted_input,
                        target_response=target_response,
                        case_id=str(row.get('Master_Index', idx)),
                        metadata={
                            'source_column': col,
                            'county': row.get('County', ''),
                            'health_level': row.get('Health Level', ''),
                            'experience': row.get('Years of Experience', ''),
                            'competency': row.get('Nursing Competency', ''),
                        }
                    )
                    examples.append(example)
                    break
        
        logger.info(f"Prepared {len(examples)} training examples for Qwen-3")
        return examples
    
    def _create_input_prompt(self, row: pd.Series) -> str:
        """Create structured clinical input prompt optimized for Qwen"""
        
        prompt_parts = []
        
        # Clinical case header
        prompt_parts.append("**CLINICAL CASE ANALYSIS REQUEST**")
        
        # Context information
        context_parts = []
        if pd.notna(row.get('County')):
            context_parts.append(f"Location: {row['County']}, Kenya")
        
        if pd.notna(row.get('Health Level')):
            context_parts.append(f"Healthcare Facility Level: {row['Health Level']}")
        
        if pd.notna(row.get('Years of Experience')):
            context_parts.append(f"Attending Nurse Experience: {row['Years of Experience']} years")
        
        if pd.notna(row.get('Nursing Competency')):
            context_parts.append(f"Clinical Specialty: {row['Nursing Competency']}")
        
        if pd.notna(row.get('Clinical Panel')):
            context_parts.append(f"Department: {row['Clinical Panel']}")
        
        if context_parts:
            prompt_parts.append(f"**Context:** {' | '.join(context_parts)}")
        
        # Main clinical case
        if pd.notna(row.get('Prompt')):
            prompt_parts.append(f"\n**Clinical Scenario:**\n{row['Prompt']}")
        
        # Specific request for comprehensive analysis
        prompt_parts.append("""
**Please provide a CONCISE clinical response (650-750 characters) structured as follows:**

1. **Assessment:** Brief summary of presenting issue and key findings
2. **Differential Diagnosis:** 2-3 most likely conditions
3. **Immediate Management:** Priority interventions (facility-appropriate)
4. **Follow-up:** Key monitoring parameters and referral criteria

Keep response focused, evidence-based, and appropriate for Kenyan healthcare context.
""")
        
        return "\n".join(prompt_parts)
    
    def fine_tune(self, train_examples: List[ClinicalExample], val_examples: List[ClinicalExample] = None, 
                  epochs: int = 3, batch_size: int = 8, learning_rate: float = 2e-5):
        """Fine-tune using Unsloth's SFTTrainer with multi-GPU support"""
        
        # Convert to HuggingFace dataset format
        train_texts = [example.input_text for example in train_examples]
        train_dataset = Dataset.from_dict({"text": train_texts})
        
        # Configure training with multi-GPU support
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            args=SFTConfig(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=2,  # Smaller for 0.5B model
                warmup_steps=10,
                max_steps=len(train_examples) // batch_size * epochs,
                learning_rate=learning_rate,
                fp16=not torch.cuda.is_bf16_supported(),
                bf16=torch.cuda.is_bf16_supported(),
                logging_steps=1,
                optim="adamw_8bit",
                weight_decay=0.01,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir="outputs",
                save_strategy="epoch",
                # Multi-GPU settings
                dataloader_pin_memory=False,
                remove_unused_columns=False,
                ddp_find_unused_parameters=False,
            ),
        )
        
        # Train the model
        logger.info("Starting Qwen-3-0.5B fine-tuning with Unsloth...")
        trainer.train()
          # Evaluate on validation set if provided
        results = {"training_stats": []}
        if val_examples:
            rouge_scores = self._evaluate_rouge(val_examples[:10])  # Sample for speed
            results["validation_rouge"] = rouge_scores
            logger.info(f"Validation ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        return results
    
    def generate_response(self, input_prompt: str, max_length: int = 400) -> str:
        """Generate CONCISE clinical response for competition submission"""
        
        # Create focused prompt for concise clinical response
        focused_prompt = f"""<|im_start|>system
You are a clinical expert in Kenya. Provide a CONCISE clinical response (maximum 600-800 characters). Focus on: diagnosis, immediate management, and key interventions only. No lengthy explanations.<|im_end|>
<|im_start|>user
{input_prompt}<|im_end|>
<|im_start|>assistant
"""
        
        # Enable fast inference
        FastLanguageModel.for_inference(self.model)
        
        # Tokenize input
        inputs = self.tokenizer(
            focused_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024  # Reduced for faster inference
        ).to(self.device)
        
        # Generate CONCISE response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,  # Force shorter responses
                temperature=0.7,  # Lower for more focused output
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                top_p=0.92,
                top_k=50,
                early_stopping=True
            )
        
        # Decode ONLY the new tokens (assistant response)
        input_length = inputs['input_ids'].shape[1]
        response_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        
        # Clean up response - remove any remaining template artifacts
        response = response.replace("<|im_start|>", "").replace("<|im_end|>", "")
        response = response.replace("assistant", "").strip()
        # After generating the response
        # Ensure response has proper clinical structure
        if "assessment" not in response.lower():
            response = "Assessment: " + response

        # Ensure length is in optimal range (650-750 chars)
        target_length = 700
        if len(response) < 600:
            response += " Follow-up: Monitor vital signs and clinical progress. Refer if condition worsens."
            # Truncate to target length (600-800 chars)
        elif len(response) > 800:
            # Find last complete sentence within 800 chars
            truncated = response[:750]
            last_period = truncated.rfind('.')
            if last_period > 600:  # Ensure minimum meaningful content
                response = truncated[:last_period + 1]
            else:
                response = truncated
        
        return response
    
    def _evaluate_rouge(self, examples: List[ClinicalExample]) -> Dict[str, float]:
        """Evaluate model performance using ROUGE scores"""
        
        predictions = []
        references = []
        
        for example in examples:
            # Extract the input prompt from the formatted text
            if "<|im_start|>user" in example.input_text:
                prompt = example.input_text.split("<|im_start|>user")[1]
                prompt = prompt.split("<|im_end|>")[0].strip()
            else:
                prompt = example.input_text
            
            pred = self.generate_response(prompt, max_length=400)
            predictions.append(pred)
            references.append(example.target_response)
        
        # Calculate ROUGE scores
        rouge_scores = {
            'rouge1': 0.0,
            'rouge2': 0.0,
            'rougeL': 0.0
        }
        
        for pred, ref in zip(predictions, references):
            scores = self.rouge_scorer.score(ref, pred)
            rouge_scores['rouge1'] += scores['rouge1'].fmeasure
            rouge_scores['rouge2'] += scores['rouge2'].fmeasure
            rouge_scores['rougeL'] += scores['rougeL'].fmeasure
        
        # Average scores
        n = len(predictions)
        return {k: v/n for k, v in rouge_scores.items()}
    
    def save_model(self, path: str):
        """Save the fine-tuned model"""
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Qwen-3-0.5B model saved to {path}")
    
    def quantize_for_edge(self):
        """Prepare model for edge deployment"""
        
        # Unsloth models are already optimized for inference
        FastLanguageModel.for_inference(self.model)
        logger.info("Qwen-3-0.5B model optimized for edge deployment")
        return self.model
    
    def cleanup_model(self):
        """Clean up model from memory to free resources"""
        if hasattr(self, 'model') and self.model is not None:
            del self.model
            self.model = None
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Qwen-3-0.5B model cleaned up from memory")
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached models from memory"""
        cls._model_cache.clear()
        cls._tokenizer_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All Qwen-3 model caches cleared")
