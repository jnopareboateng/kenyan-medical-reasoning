"""
Llama-3.2-3B-Instruct Implementation for Clinical Response Generation
Using Meta's latest 3B parameter instruction-tuned model with Unsloth optimization
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

logger = CompetitionLogger("Llama32Model")
paths = get_project_paths()

class ClinicalLlama32Model:
    """Llama-3.2-3B-Instruct fine-tuned for clinical reasoning with Unsloth optimization"""
    
    def __init__(self, model_name: str = "meta-llama/Llama-3.2-3B-Instruct", load_in_4bit: bool = True):
        """Initialize Llama-3.2-3B with Unsloth for 4-bit quantization"""
        
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_in_4bit = load_in_4bit
        
        logger.info(f"Loading {model_name} with Unsloth optimization")
        
        # Load model with Unsloth for efficient training and inference
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,  # Llama-3.2 supports up to 8K, but 2K is sufficient for clinical cases
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
        
        # Llama-3.2 chat template
        self.chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert clinical practitioner specializing in healthcare delivery in Kenya. You have extensive experience working within the Kenyan healthcare system and understand the unique challenges, resource constraints, and cultural considerations. Provide detailed, evidence-based clinical assessments and management plans that are appropriate for the local context.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""
        
        logger.info(f"Llama-3.2-3B loaded with {sum(p.numel() for p in self.model.parameters())} parameters")
    
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
                    
                    # Format for Llama-3.2 chat template
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
        
        logger.info(f"Prepared {len(examples)} training examples for Llama-3.2")
        return examples
    
    def _create_input_prompt(self, row: pd.Series) -> str:
        """Create structured clinical input prompt optimized for Llama-3.2"""
        
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
**Please provide a comprehensive clinical response including:**

1. **Assessment & Differential Diagnosis:** Based on clinical presentation
2. **Immediate Management:** Priority interventions and stabilization
3. **Diagnostic Approach:** Investigations appropriate for the facility level
4. **Treatment Plan:** Evidence-based management considering Kenyan guidelines
5. **Patient Education:** Key counseling points for patient/family
6. **Follow-up & Referral:** Monitoring plan and when to escalate care

Consider resource constraints, local disease patterns, and cultural factors relevant to healthcare delivery in Kenya.
""")
        
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
                warmup_steps=15,
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
            ),
        )
        
        # Train the model
        logger.info("Starting Llama-3.2-3B fine-tuning with Unsloth...")
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
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert clinical practitioner specializing in healthcare delivery in Kenya. You have extensive experience working within the Kenyan healthcare system and understand the unique challenges, resource constraints, and cultural considerations. Provide detailed, evidence-based clinical assessments and management plans that are appropriate for the local context.<|eot_id|><|start_header_id|>user<|end_header_id|>

{input_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        # Enable fast inference
        FastLanguageModel.for_inference(self.model)
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1536
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9,
                top_k=50
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        # Remove any trailing tokens
        if "<|eot_id|>" in response:
            response = response.split("<|eot_id|>")[0].strip()
        
        return response
    
    def _evaluate_rouge(self, examples: List[ClinicalExample]) -> Dict[str, float]:
        """Evaluate model performance using ROUGE scores"""
        
        predictions = []
        references = []
        
        for example in examples:
            # Extract the input prompt from the formatted text
            if "<|start_header_id|>user<|end_header_id|>" in example.input_text:
                prompt = example.input_text.split("<|start_header_id|>user<|end_header_id|>")[1]
                prompt = prompt.split("<|eot_id|>")[0].strip()
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
        logger.info(f"Llama-3.2-3B model saved to {path}")
    
    def quantize_for_edge(self):
        """Prepare model for edge deployment"""
        
        # Unsloth models are already optimized for inference
        FastLanguageModel.for_inference(self.model)
        logger.info("Llama-3.2-3B model optimized for edge deployment")
        return self.model
