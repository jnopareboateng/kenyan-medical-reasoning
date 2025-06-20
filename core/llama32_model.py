"""
Llama-3.2-3B-Instruct Implementation for Clinical Response Generation
Using Meta's latest 3B parameter instruction-tuned model with Unsloth optimization
"""

import torch
import pandas as pd
from unsloth import FastLanguageModel
from typing import List, Dict

from core.base_model import BaseUnslothModel
from core.ml_model import ClinicalExample
from utils.logger import CompetitionLogger

logger = CompetitionLogger("Llama32Model")

class ClinicalLlama32Model(BaseUnslothModel):
    """Llama-3.2-3B-Instruct fine-tuned for clinical reasoning with Unsloth optimization"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert clinical practitioner specializing in healthcare delivery in Kenya. You have extensive experience working within the Kenyan healthcare system and understand the unique challenges, resource constraints, and cultural considerations. You think step by step internally without showing it to the user ONLY produce a final detailed, evidence-based clinical assessments and management plans that are appropriate for the local context. Your response must be EXACTLY the following structure:
1. Assessment: [Brief summary of the presenting issue and key findings]
2. Differential Diagnosis: [List 2-3 likely diagnoses]
3. Immediate Management: [Detail critical interventions appropriate to the setting]
4. Follow-up: [Outline essential monitoring and referral criteria]
Ensure your final response is evidence-based, focused on local resource constraints, and between 650-750 characters.<|eot_id|><|start_header_id|>user<|end_header_id|>

{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{response}<|eot_id|>"""
        logger.info(f"Llama-3.2-3B loaded with {sum(p.numel() for p in self.model.parameters())} parameters")

    def prepare_training_data(self, df: pd.DataFrame) -> List[ClinicalExample]:
        """Convert DataFrame to training examples using expert responses"""
        
        examples = []
        expert_columns = ['Nursing Competency', 'Clinical Panel', 'Clinician', 'GPT4.0', 'LLAMA', 'GEMINI']
        
        for idx, row in df.iterrows():
            input_text = self._create_input_prompt(row)
            
            for col in expert_columns:
                if col in df.columns and pd.notna(row[col]) and len(str(row[col]).strip()) > 50:
                    target_response = str(row[col]).strip()
                    
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
        
        prompt_parts.append("**CLINICAL CASE ANALYSIS REQUEST**")
        
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
        
        if pd.notna(row.get('Prompt')):
            prompt_parts.append(f"\n**Clinical Scenario:**\n{row['Prompt']}")
        
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

    def generate_response(self, input_prompt: str, max_length: int = 512) -> str:
        """Generate clinical response using the fine-tuned model"""
        
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are an expert clinical practitioner specializing in healthcare delivery in Kenya. You have extensive experience working within the Kenyan healthcare system and understand the unique challenges, resource constraints, and cultural considerations.Internally consider all necessary steps and evidence, but DO NOT output your reasoning. 
Your final answer must strictly follow this format:
1. ASSESSMENT: [summary]
2. DIFFERENTIAL: [2-3 likely diagnoses]
3. MANAGEMENT: [immediate actions]
4. FOLLOW-UP: [monitoring/referral instructions]
Ensure your final response is between 650-750 characters and concise.<|eot_id|><|start_header_id|>user<|end_header_id|>

{input_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        
        FastLanguageModel.for_inference(self.model)
        
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.4,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
                top_p=0.9,
                top_k=50
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        
        if "<|eot_id|>" in response:
            response = response.split("<|eot_id|>")[0].strip()
        
        return response

    def _extract_prompt_from_chat_template(self, text: str) -> str:
        if "<|start_header_id|>user<|end_header_id|>" in text:
            prompt = text.split("<|start_header_id|>user<|end_header_id|>")[1]
            prompt = prompt.split("<|eot_id|>")[0].strip()
            return prompt
        return text
