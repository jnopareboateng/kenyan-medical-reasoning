"""
Qwen-3-0.5B Implementation for Clinical Response Generation
Using Alibaba's latest 0.5B parameter instruction-tuned model with Unsloth optimization
"""

import torch
import pandas as pd
from unsloth import FastLanguageModel
from typing import List, Dict

from core.base_model import BaseUnslothModel
from core.ml_model import ClinicalExample
from utils.logger import CompetitionLogger

logger = CompetitionLogger("Qwen3Model")

class ClinicalQwen3Model(BaseUnslothModel):
    """Qwen-3-0.5B fine-tuned for clinical reasoning with Unsloth optimization"""

    def __init__(self, config: Dict):
        super().__init__(config)
        self.chat_template = """<|im_start|>system
You are Qwen, an expert clinical practitioner specializing in healthcare delivery in Kenya. 
Internally, think step by step and engage in chain-of-thought reasoning to analyze the case, but do not output this reasoning.
ONLY produce a final, concise clinical response in EXACTLY the following structure:
1. Assessment: [Brief summary of the presenting issue and key findings]
2. Differential Diagnosis: [List 2-3 likely diagnoses]
3. Immediate Management: [Detail critical interventions appropriate to the setting]
4. Follow-up: [Outline essential monitoring and referral criteria]
Ensure your final response is evidence-based, focused on local resource constraints, and between 650-750 characters.<|im_end|>
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
        
        logger.info(f"Prepared {len(examples)} training examples for Qwen-3")
        return examples

    def _create_input_prompt(self, row: pd.Series) -> str:
        """Create structured clinical input prompt optimized for Qwen"""
        
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
**Please provide a CONCISE clinical response (650-750 characters) structured as follows:**

1. **Assessment:** Brief summary of presenting issue and key findings
2. **Differential Diagnosis:** 2-3 most likely conditions
3. **Immediate Management:** Priority interventions (facility-appropriate)
4. **Follow-up:** Key monitoring parameters and referral criteria

Keep response focused, evidence-based, and appropriate for Kenyan healthcare context.
""")
        
        return "\n".join(prompt_parts)

    def generate_response(self, input_prompt: str, max_length: int = 400) -> str:
        """Generate CONCISE clinical response for competition submission"""
        
        focused_prompt = f"""<|im_start|>system
You are a clinical expert in Kenya. Internally consider all necessary steps and evidence, but DO NOT output your reasoning. 
Your final answer must strictly follow this format:
1. ASSESSMENT: [summary]
2. DIFFERENTIAL: [2-3 likely diagnoses]
3. MANAGEMENT: [immediate actions]
4. FOLLOW-UP: [monitoring/referral instructions]
Ensure your final response is between 650-750 characters and concise.<|im_end|>
<|im_start|>user
{input_prompt}<|im_end|>
<|im_start|>assistant
"""
        
        FastLanguageModel.for_inference(self.model)
        
        inputs = self.tokenizer(
            focused_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=0.4,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,
                top_p=0.92,
                top_k=50,
                early_stopping=True
            )
        
        input_length = inputs['input_ids'].shape[1]
        response_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
        
        response = response.replace("<|im_start|>", "").replace("<|im_end|>", "")
        response = response.replace("assistant", "").strip()

        if "assessment" not in response.lower():
            response = "Assessment: " + response

        if len(response) < 600:
            response += " Follow-up: Monitor vital signs and clinical progress. Refer if condition worsens."
        elif len(response) > 800:
            truncated = response[:750]
            last_period = truncated.rfind('.')
            if last_period > 600:
                response = truncated[:last_period + 1]
            else:
                response = truncated
        
        return response

    def _extract_prompt_from_chat_template(self, text: str) -> str:
        if "<|im_start|>user" in text:
            prompt = text.split("<|im_start|>user")[1]
            prompt = prompt.split("<|im_end|>")[0].strip()
            return prompt
        return text
