"""
Phi-4-mini-instruct Implementation for Clinical Response Generation
Using Microsoft's latest 3.8B parameter model with Unsloth optimization
"""

import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from unsloth import FastLanguageModel
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

logger = CompetitionLogger("Phi4Model")
paths = get_project_paths()

class ClinicalPhi4Model:
    """Phi-4-mini-instruct fine-tuned for clinical reasoning with Unsloth optimization"""
    
    def __init__(self, model_name: str = "microsoft/Phi-4-mini-instruct", load_in_4bit: bool = True):
        """Initialize Phi-4-mini with Unsloth for 4-bit quantization"""
        
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_in_4bit = load_in_4bit
        
        logger.info(f"Loading {model_name} with Unsloth optimization")
        
        # Load model with Unsloth for efficient training and inference
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,  # Phi-4-mini supports up to 128K, but 2K is sufficient for clinical cases
            dtype=None,  # Auto-detect
            load_in_4bit=load_in_4bit,  # Use 4-bit quantization for efficiency
        )
        
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
        
        # Chat template for Phi-4
        self.chat_template = """<|system|>You are an expert clinical practitioner in Kenya. Provide detailed, culturally appropriate medical assessments and management plans.<|end|>
<|user|>{prompt}<|end|>
<|assistant|>{response}<|end|>"""
        
        logger.info(f"Phi-4-mini loaded with {sum(p.numel() for p in self.model.parameters())} parameters")
    
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
                    
                    # Format for Phi-4 chat template
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
                        }
                    )
                    examples.append(example)
                    break
        
        logger.info(f"Prepared {len(examples)} training examples from expert responses")
        return examples
    
    def _create_input_prompt(self, row: pd.Series) -> str:
        """Create structured clinical input prompt"""
        
        prompt_parts = []
        
        # Add context information
        if pd.notna(row.get('County')):
            prompt_parts.append(f"Location: {row['County']}, Kenya")
        
        if pd.notna(row.get('Health Level')):
            prompt_parts.append(f"Healthcare Facility: {row['Health Level']}")
        
        if pd.notna(row.get('Years of Experience')):
            prompt_parts.append(f"Nurse Experience: {row['Years of Experience']} years")
        
        # Add main clinical case
        if pd.notna(row.get('Prompt')):
            prompt_parts.append(f"\nClinical Case:\n{row['Prompt']}")
        
        # Add specific competency focus if available
        if pd.notna(row.get('Nursing Competency')):
            prompt_parts.append(f"\nFocus Area: {row['Nursing Competency']}")
        
        prompt_parts.append("\nProvide a comprehensive clinical assessment and management plan appropriate for the Kenyan healthcare context:")
        
        return "\n".join(prompt_parts)
    
    def fine_tune(self, train_examples: List[ClinicalExample], val_examples: List[ClinicalExample] = None, 
                  epochs: int = 3, batch_size: int = 4, learning_rate: float = 2e-5):
        """Fine-tune using Unsloth's SFTTrainer"""
        
        # Convert to HuggingFace dataset format
        train_texts = [example.input_text for example in train_examples]
        train_dataset = Dataset.from_dict({"text": train_texts})
        
        # Configure training
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=train_dataset,
            args=SFTConfig(
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=4,
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
            ),
        )
        
        # Train the model
        logger.info("Starting Phi-4-mini fine-tuning with Unsloth...")
        trainer.train()
        
        # Evaluate on validation set if provided
        results = {"training_stats": []}
        if val_examples:
            rouge_scores = self._evaluate_rouge(val_examples[:10])  # Sample for speed
            results["validation_rouge"] = rouge_scores
            logger.info(f"Validation ROUGE-L: {rouge_scores['rougeL']:.4f}")
        
        return results
    
    def generate_response(self, input_prompt: str, max_length: int = 512) -> str:
        """Generate clinical response using the fine-tuned model"""
        
        # Format input for chat template
        formatted_prompt = f"<|system|>You are an expert clinical practitioner in Kenya. Provide detailed, culturally appropriate medical assessments and management plans.<|end|>\n<|user|>{input_prompt}<|end|>\n<|assistant|>"
        
        # Enable fast inference
        FastLanguageModel.for_inference(self.model)
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        
        return response
    
    def _evaluate_rouge(self, examples: List[ClinicalExample]) -> Dict[str, float]:
        """Evaluate model performance using ROUGE scores"""
        
        predictions = []
        references = []
        
        for example in examples:
            # Extract the input prompt from the formatted text
            if "<|user|>" in example.input_text and "<|end|>" in example.input_text:
                prompt = example.input_text.split("<|user|>")[1].split("<|end|>")[0]
            else:
                prompt = example.input_text
            
            pred = self.generate_response(prompt, max_length=200)
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
        logger.info(f"Phi-4-mini model saved to {path}")
    
    def quantize_for_edge(self):
        """Prepare model for edge deployment"""
        
        # Unsloth models are already optimized for inference
        FastLanguageModel.for_inference(self.model)
        logger.info("Phi-4-mini model optimized for edge deployment")
        return self.model
