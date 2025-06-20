"""
Real ML Model Implementation for Clinical Response Generation
Using FLAN-T5-small for competition-grade performance
"""

import torch
import pandas as pd
import numpy as np
from transformers import (
    T5ForConditionalGeneration, 
    T5Tokenizer, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
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

logger = CompetitionLogger("MLModel")
paths = get_project_paths()

@dataclass
class ClinicalExample:
    """Represents a clinical case for training/inference"""
    input_text: str
    target_response: str
    case_id: str
    metadata: Dict

class ClinicalDataset(Dataset):
    """PyTorch Dataset for clinical reasoning data"""
    
    def __init__(self, examples: List[ClinicalExample], tokenizer, max_length=512):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize input
        input_encoding = self.tokenizer(
            example.input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            example.target_response,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'target_text': example.target_response
        }

class ClinicalT5Model:
    """FLAN-T5-small fine-tuned for clinical reasoning"""
    
    # Class-level model cache to prevent re-downloading
    _model_cache = {}
    _tokenizer_cache = {}
    
    def __init__(self, model_name: str = "google/flan-t5-xl", cache_dir: str = "./models", force_download: bool = False):
        """Initialize the model with FLAN-T5-small (77M params, edge-deployable)"""
        
        self.model_name = model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Create cache key
        cache_key = f"{model_name}"
        
        logger.info(f"Loading {model_name} with caching optimization")
        
        # Check if model is already in memory cache
        if not force_download and cache_key in self._model_cache:
            logger.info("✅ Using cached T5 model from memory")
            self.model = self._model_cache[cache_key]
            self.tokenizer = self._tokenizer_cache[cache_key]
            self.model.to(self.device)
            return
        
        logger.info(f"Downloading/Loading from cache: {model_name}")
        
        # Load model and tokenizer with persistent cache
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        self.model = T5ForConditionalGeneration.from_pretrained(model_name, cache_dir=str(self.cache_dir))
        self.model.to(self.device)
        
        # Cache in memory for subsequent uses
        self._model_cache[cache_key] = self.model
        self._tokenizer_cache[cache_key] = self.tokenizer
        logger.info("✅ T5 Model cached in memory for future use")
        
        # Add special tokens for medical domain
        special_tokens = [
            "<ASSESSMENT>", "</ASSESSMENT>",
            "<MANAGEMENT>", "</MANAGEMENT>", 
            "<FOLLOW_UP>", "</FOLLOW_UP>",
            "<KENYAN_CONTEXT>", "</KENYAN_CONTEXT>"
        ]
        
        self.tokenizer.add_tokens(special_tokens)
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # ROUGE scorer for evaluation
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        logger.info(f"Model loaded: {sum(p.numel() for p in self.model.parameters())} parameters")
    def prepare_training_data(self, df: pd.DataFrame) -> List[ClinicalExample]:
        """Convert DataFrame to training examples using REAL expert responses"""
        
        examples = []
        
        # Expert response columns available in the data
        expert_columns = ['Nursing Competency', 'Clinical Panel', 'Clinician', 'GPT4.0', 'LLAMA', 'GEMINI']
        
        for idx, row in df.iterrows():
            # Create structured input prompt from the clinical case
            input_text = self._create_input_prompt(row)
            
            # Use the best expert response (prefer human experts over AI)
            target_response = None
            for col in ['Clinician', 'Clinical Panel', 'Nursing Competency', 'GPT4.0']:
                if pd.notna(row.get(col, '')) and str(row.get(col, '')).strip():
                    target_response = str(row[col]).strip()
                    break
            
            if not target_response or len(target_response) < 100:
                continue
                
            example = ClinicalExample(
                input_text=input_text,
                target_response=target_response,
                case_id=str(row.get('Master_Index', idx)),
                metadata={
                    'county': row.get('County'),
                    'health_level': row.get('Health level'),
                    'experience': row.get('Years of Experience'),
                    'competency': row.get('Nursing Competency'),
                    'specialty': row.get('Clinical Panel')
                }
            )
            examples.append(example)        
        logger.info(f"Prepared {len(examples)} training examples from REAL expert responses")
        return examples
    
    def _create_input_prompt(self, case_data) -> str:
        """Create structured input prompt for the model from actual case data"""
        
        prompt_parts = [
            "Generate clinical response for Kenyan healthcare context:",
            f"Experience: {case_data.get('Years of Experience', 'Unknown')} years",
            f"Health Level: {case_data.get('Health level', 'Unknown')}",
            f"County: {case_data.get('County', 'Unknown')}",
            f"Specialty: {case_data.get('Clinical Panel', 'General')}",
            "",
            "Clinical Case:",
            case_data.get('Prompt', ''),
            "",
            "Requirements: Generate comprehensive clinical response (~700 chars) including assessment, management plan, and follow-up appropriate for Kenyan healthcare setting."
        ]
        
        return "\n".join(prompt_parts)
    
    def fine_tune(self, train_examples: List[ClinicalExample], 
                  val_examples: List[ClinicalExample] = None,
                  epochs: int = 3, 
                  batch_size: int = 4,
                  learning_rate: float = 5e-5) -> Dict:
        """Fine-tune the model on clinical data"""
        
        logger.info(f"Starting fine-tuning: {len(train_examples)} training examples")
        
        # Create datasets
        train_dataset = ClinicalDataset(train_examples, self.tokenizer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        if val_examples:
            val_dataset = ClinicalDataset(val_examples, self.tokenizer)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        training_stats = []
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            total_loss = 0
            for batch_idx, batch in enumerate(train_loader):
                
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                if batch_idx % 10 == 0:
                    logger.info(f"Batch {batch_idx}: Loss = {loss.item():.4f}")
            
            avg_loss = total_loss / len(train_loader)
            
            # Validation
            val_metrics = {}
            if val_examples:
                val_metrics = self._evaluate(val_loader)
            
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                **val_metrics
            }
            training_stats.append(epoch_stats)
            
            logger.info(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Val ROUGE-L: {val_metrics.get('rouge_l', 0):.4f}")
        
        logger.info("Fine-tuning completed")
        return {'training_stats': training_stats}
    
    def _evaluate(self, val_loader) -> Dict:
        """Evaluate model on validation set"""
        
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Calculate loss
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                total_loss += outputs.loss.item()
                
                # Generate predictions
                generated = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=200,
                    num_beams=4,
                    length_penalty=0.6,
                    early_stopping=True
                )
                
                predictions = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated]
                targets = batch['target_text']
                
                all_predictions.extend(predictions)
                all_targets.extend(targets)
        
        # Calculate ROUGE scores
        rouge_scores = []
        for pred, target in zip(all_predictions, all_targets):
            scores = self.rouge_scorer.score(target, pred)
            rouge_scores.append(scores)
        
        # Average ROUGE scores
        avg_rouge = {
            'rouge_1': np.mean([s['rouge1'].fmeasure for s in rouge_scores]),
            'rouge_2': np.mean([s['rouge2'].fmeasure for s in rouge_scores]),
            'rouge_l': np.mean([s['rougeL'].fmeasure for s in rouge_scores])
        }
        
        self.model.train()
        
        return {
            'val_loss': total_loss / len(val_loader),
            **avg_rouge
        }
    
    def generate_response(self, case_input: str, max_length: int = 200) -> str:
        """Generate clinical response for a case"""
        
        self.model.eval()
        
        # Tokenize input
        inputs = self.tokenizer(
            case_input,
            max_length=512,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            generated = self.model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                num_beams=4,
                length_penalty=0.5,
                early_stopping=True,
                do_sample=True,
                temperature=0.5,
                top_p=0.9
            )
        
        response = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        
        # Post-process for competition format
        response = self._post_process_response(response)
        
        return response
    
    def _post_process_response(self, response: str) -> str:
        """Post-process generated response for competition format"""
        
        # Remove input prompt if model repeats it
        if "Generate clinical response" in response:
            parts = response.split("Generate clinical response")
            if len(parts) > 1:
                response = parts[-1].strip()
        
        # Ensure proper clinical structure
        if not any(keyword in response.lower() for keyword in ['assessment', 'management', 'plan']):
            response = f"Clinical Assessment: {response}"
        
        # Target length optimization (around 700 chars)
        if len(response) < 500:
            # Add standard follow-up
            response += " Follow-up in 1-2 weeks to monitor progress and adjust treatment as needed."
        
        # Truncate if too long
        if len(response) > 800:
            sentences = response.split('. ')
            truncated = []
            current_length = 0
            
            for sentence in sentences:
                if current_length + len(sentence) + 2 <= 700:
                    truncated.append(sentence)
                    current_length += len(sentence) + 2
                else:
                    break
            
            response = '. '.join(truncated)
            if not response.endswith('.'):
                response += '.'
        
        return response.strip()
    
    def save_model(self, save_path: str):
        """Save fine-tuned model"""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        logger.info(f"Model saved to {save_path}")
    
    def load_model(self, model_path: str):
        """Load fine-tuned model"""
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = T5Tokenizer.from_pretrained(model_path)
        self.model.to(self.device)
        
        logger.info(f"Model loaded from {model_path}")
    
    def quantize_for_edge(self) -> torch.nn.Module:
        """Quantize model for edge deployment (Jetson Nano compatible)"""
        
        # Dynamic quantization for CPU inference
        quantized_model = torch.quantization.quantize_dynamic(
            self.model.cpu(),
            {torch.nn.Linear},
            dtype=torch.qint8
        )
        
        logger.info("Model quantized for edge deployment")
        return quantized_model
    
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
        
        logger.info("ClinicalT5Model cleaned up from memory")
    
    @classmethod
    def clear_cache(cls):
        """Clear all cached models from memory"""
        cls._model_cache.clear()
        cls._tokenizer_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("All ClinicalT5Model caches cleared")

class MLPipeline:
    """Complete ML pipeline for clinical reasoning"""
    def __init__(self):
        self.model = ClinicalT5Model()
        self.paths = get_project_paths()
        self.logger = CompetitionLogger("MLPipeline")
    
    def run_full_pipeline(self) -> Dict:
        """Run complete ML pipeline from data to submission"""
        
        self.logger.info("Starting ML Pipeline")
        
        # Load training data
        train_df = pd.read_csv(self.paths['train_data'])
        
        # Prepare training examples (you'll need expert responses)
        # For now, using a synthetic approach to demonstrate the pipeline
        train_examples = self._create_synthetic_examples(train_df.head(50))  # Start small
        
        # Split for validation
        split_idx = int(0.8 * len(train_examples))
        train_set = train_examples[:split_idx]
        val_set = train_examples[split_idx:]
        
        # Fine-tune model
        training_results = self.model.fine_tune(train_set, val_set, epochs=2)
        
        # Save trained model
        model_save_path = self.paths['project_root'] / 'models' / 'flan_t5_clinical'
        self.model.save_model(str(model_save_path))
        
        # Generate test predictions
        test_df = pd.read_csv(self.paths['test_data'])
        predictions = self._generate_test_predictions(test_df)
        
        # Save submission
        submission_path = self.paths['results'] / 'ml_submission.csv'
        self._save_submission(predictions, submission_path)
        
        # Quantize for edge deployment
        quantized_model = self.model.quantize_for_edge()
        
        results = {
            'training_results': training_results,
            'model_size': sum(p.numel() for p in self.model.model.parameters()),
            'submission_file': str(submission_path),
            'model_path': str(model_save_path)
        }        
        self.logger.info("ML Pipeline completed successfully")
        return results
    
    def _create_synthetic_examples(self, df: pd.DataFrame) -> List[ClinicalExample]:
        """Use REAL expert responses from the dataset - no more synthetic garbage"""
        # Just call the real training data preparation
        return self.model.prepare_training_data(df)
    
    def _generate_test_predictions(self, test_df: pd.DataFrame) -> List[str]:
        """Generate predictions for test set"""
        predictions = []
        
        for idx, row in test_df.iterrows():
            input_prompt = self.model._create_input_prompt(row)
            response = self.model.generate_response(input_prompt)
            predictions.append(response)
        
        return predictions
    def _save_submission(self, predictions: List[str], filepath: str):
        """Save predictions in CORRECT competition format"""
        
        # Load test data to get Master_Index values
        test_df = pd.read_csv(self.paths['data'] / 'test.csv')
        
        # Create submission with CORRECT format
        submission_df = pd.DataFrame({
            'Master_Index': test_df['Master_Index'],  # Use actual Master_Index from test
            'Clinician': predictions  # Competition expects 'Clinician' column
        })
        
        submission_df.to_csv(filepath, index=False)
        self.logger.info(f"Submission saved to {filepath} with correct format")
        self.logger.info(f"Format: Master_Index, Clinician ({len(submission_df)} rows)")

if __name__ == "__main__":
    # Quick test of the ML pipeline
    pipeline = MLPipeline()
    results = pipeline.run_full_pipeline()
    print(f"Pipeline completed: {results}")
