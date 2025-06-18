"""
Meditron-7B Implementation for Clinical Response Generation
Using EPFL's medical-specialized 7B parameter model with Unsloth optimization
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

logger = CompetitionLogger("MeditronModel")
paths = get_project_paths()

class ClinicalMeditronModel:
    """Meditron-7B fine-tuned for clinical reasoning with Unsloth optimization"""
    
    # Class-level model cache to prevent re-downloading
    _model_cache = {}
    _tokenizer_cache = {}
    
    def __init__(self, model_name: str = "epfl-llm/meditron-7b", load_in_4bit: bool = True, 
                 cache_dir: str = "./models", force_download: bool = False):
        """Initialize Meditron-7B with Unsloth for 4-bit quantization"""
        
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
            max_seq_length=2048,  # Meditron supports 2K context
            dtype=None,  # Auto-detect
            load_in_4bit=load_in_4bit,  # Use 4-bit quantization for efficiency
            cache_dir=str(self.cache_dir),  # Use persistent cache
        )
        
        # Cache in memory for subsequent uses
        self._model_cache[cache_key] = self.model
        self._tokenizer_cache[cache_key] = self.tokenizer
        logger.info("✅ Model cached in memory for future use")
        
        # Configure for fine-tuning with LoRA (Meditron uses Llama2 architecture)
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
        
        # Medical instruction template for Meditron
        self.instruction_template = """### Instruction:
You are a medical expert specializing in clinical reasoning for healthcare in Kenya. Analyze the following clinical case and provide a comprehensive assessment and management plan appropriate for the local healthcare context.

### Clinical Case:
{prompt}

### Response:
{response}"""
        
        logger.info(f"Meditron-7B loaded with {sum(p.numel() for p in self.model.parameters())} parameters")
    
    def prepare_training_data(self, df: pd.DataFrame) -> List[ClinicalExample]:
        """Convert DataFrame to training examples using expert medical responses"""
        
        examples = []
        expert_columns = ['Nursing Competency', 'Clinical Panel', 'Clinician', 'GPT4.0', 'LLAMA', 'GEMINI']
        
        for idx, row in df.iterrows():
            input_text = self._create_clinical_prompt(row)
            
            # Prioritize actual clinician responses for medical model
            priority_columns = ['Clinician', 'Clinical Panel', 'Nursing Competency', 'GPT4.0', 'LLAMA', 'GEMINI']
            
            for col in priority_columns:
                if col in df.columns and pd.notna(row[col]) and len(str(row[col]).strip()) > 50:
                    target_response = str(row[col]).strip()
                    
                    # Format for medical instruction template
                    formatted_input = self.instruction_template.format(
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
                            'clinical_panel': row.get('Clinical Panel', ''),
                        }
                    )
                    examples.append(example)
                    break
        
        logger.info(f"Prepared {len(examples)} medical training examples")
        return examples
    
    def _create_clinical_prompt(self, row: pd.Series) -> str:
        """Create structured medical clinical input prompt"""
        
        prompt_parts = []
        
        # Patient demographics and context
        if pd.notna(row.get('County')):
            prompt_parts.append(f"**Location:** {row['County']}, Kenya")
        
        if pd.notna(row.get('Health Level')):
            prompt_parts.append(f"**Healthcare Facility:** {row['Health Level']}")
        
        if pd.notna(row.get('Years of Experience')):
            prompt_parts.append(f"**Attending Nurse Experience:** {row['Years of Experience']} years")
        
        # Clinical specialty context
        if pd.notna(row.get('Nursing Competency')):
            prompt_parts.append(f"**Clinical Specialty:** {row['Nursing Competency']}")
        
        if pd.notna(row.get('Clinical Panel')):
            prompt_parts.append(f"**Department:** {row['Clinical Panel']}")
        
        # Main clinical case presentation
        if pd.notna(row.get('Prompt')):
            prompt_parts.append(f"\n**Clinical Presentation:**\n{row['Prompt']}")
        
        # Medical reasoning request
        prompt_parts.append("""
**Please provide:**
1. **Clinical Assessment:** Key findings and differential diagnosis
2. **Immediate Management:** Urgent interventions and stabilization
3. **Diagnostic Workup:** Appropriate investigations within resource constraints
4. **Treatment Plan:** Evidence-based management suitable for Kenya
5. **Follow-up:** Monitoring and referral considerations
""")
        
        return "\n".join(prompt_parts)
    
    def fine_tune(self, train_examples: List[ClinicalExample], val_examples: List[ClinicalExample] = None, 
                  epochs: int = 3, batch_size: int = 2, learning_rate: float = 2e-5):
        """Fine-tune Meditron using Unsloth's SFTTrainer"""
        
        # Convert to HuggingFace dataset format
        train_texts = [example.input_text for example in train_examples]
        train_dataset = Dataset.from_dict({"text": train_texts})
        
        # Configure training (smaller batch size for 7B model)
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            args=SFTConfig(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=8,  # Larger accumulation for 7B model
                warmup_steps=20,
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
                dataloader_num_workers=0,  # Prevent multiprocessing issues
            ),
        )
        
        # Train the medical model
        logger.info("Starting Meditron-7B fine-tuning with Unsloth...")
        trainer.train()
        
        # Evaluate on validation set if provided
        results = {"training_stats": []}
        if val_examples:
            rouge_scores = self._evaluate_medical_rouge(val_examples[:5])  # Smaller sample for 7B model
            results["validation_rouge"] = rouge_scores
            logger.info(f"Medical Validation ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        return results
    
    def generate_response(self, input_prompt: str, max_length: int = 512) -> str:
        """Generate clinical response using the fine-tuned Meditron model"""
        
        # Format input for medical instruction template
        formatted_prompt = f"""### Instruction:
You are a medical expert specializing in clinical reasoning for healthcare in Kenya. Analyze the following clinical case and provide a comprehensive assessment and management plan appropriate for the local healthcare context.

### Clinical Case:
{input_prompt}

### Response:
"""
        
        # Enable fast inference
        FastLanguageModel.for_inference(self.model)
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536  # Leave room for response
        ).to(self.device)
        
        # Generate response with medical-appropriate parameters
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.6,  # Lower temperature for medical accuracy
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.05,
                top_p=0.9
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the response part
        if "### Response:" in response:
            response = response.split("### Response:")[-1].strip()
        
        return response
    
    def _evaluate_medical_rouge(self, examples: List[ClinicalExample]) -> Dict[str, float]:
        """Evaluate medical model performance using ROUGE scores"""
        
        predictions = []
        references = []
        
        for example in examples:
            # Extract the clinical prompt from the formatted text
            if "### Clinical Case:" in example.input_text and "### Response:" in example.input_text:
                prompt = example.input_text.split("### Clinical Case:")[1].split("### Response:")[0].strip()
            else:
                prompt = example.input_text
            
            pred = self.generate_response(prompt, max_length=300)
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
        """Save the fine-tuned medical model"""
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info(f"Meditron-7B model saved to {path}")
    
    def quantize_for_edge(self):
        """Prepare medical model for edge deployment"""
        
        # Unsloth models are already optimized for inference
        FastLanguageModel.for_inference(self.model)
        logger.info("Meditron-7B model optimized for edge deployment")
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
        
        logger.info("Meditron-7B model cleaned up from memory")
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached models from memory"""
        cls._model_cache.clear()
        cls._tokenizer_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All Meditron model caches cleared")
