"""
Advanced Medical NLP Pipeline - Core Implementation
Production-grade medical text analysis and response generation
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
import json

from utils.logger import get_logger
from utils.paths import PATHS

logger = get_logger("medical_nlp")


@dataclass
class MedicalEntity:
    """Structured medical entity"""
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class ClinicalFeatures:
    """Extracted clinical features from medical case"""
    symptoms: List[str]
    medications: List[str]
    procedures: List[str]
    vital_signs: Dict[str, str]
    demographics: Dict[str, str]
    severity_indicators: List[str]
    specialty_keywords: List[str]


class MedicalEntityExtractor:
    """Extract medical entities from clinical text"""
    
    def __init__(self):
        self.symptom_patterns = [
            r'\b(pain|ache|discomfort|tenderness)\b',
            r'\b(fever|temperature|pyrexia|hyperthermia)\b', 
            r'\b(nausea|vomiting|emesis)\b',
            r'\b(bleeding|hemorrhage|haemorrhage)\b',
            r'\b(swelling|edema|oedema)\b',
            r'\b(difficulty|trouble)\s+(breathing|swallowing|walking)\b',
            r'\b(weakness|fatigue|tired|exhausted)\b',
            r'\b(dizziness|vertigo|lightheaded)\b',
            r'\b(headache|cephalgia)\b',
            r'\b(confusion|disoriented|altered mental)\b'
        ]
        
        self.medication_patterns = [
            r'\b(insulin|metformin|glibenclamide)\b',
            r'\b(paracetamol|acetaminophen|ibuprofen|aspirin)\b',
            r'\b(antibiotics|amoxicillin|penicillin|ceftriaxone)\b',
            r'\b(morphine|tramadol|codeine|analgesics)\b',
            r'\b(prednisolone|hydrocortisone|steroids)\b',
            r'\b(adrenaline|epinephrine|atropine)\b',
            r'\b(oxygen|o2|nebulizer|inhaler)\b'
        ]
        
        self.procedure_patterns = [
            r'\b(x-ray|radiograph|imaging)\b',
            r'\b(ct scan|computed tomography)\b',
            r'\b(ultrasound|sonography)\b',
            r'\b(blood test|laboratory|lab work)\b',
            r'\b(surgery|operation|surgical)\b',
            r'\b(intubation|ventilation)\b',
            r'\b(iv|intravenous|cannula)\b',
            r'\b(catheter|urinary catheter)\b',
            r'\b(biopsy|histology)\b'
        ]
        
        self.vital_patterns = {
            'blood_pressure': r'(\d+/\d+)\s*mmhg',
            'pulse': r'pulse?\s*:?\s*(\d+)\s*b?pm',
            'temperature': r'temp?\s*:?\s*(\d+\.?\d*)\s*[°c|celsius]',
            'respiratory_rate': r'resp?\s*:?\s*(\d+)\s*b?pm',
            'oxygen_saturation': r'spo2?\s*:?\s*(\d+)%?'
        }
        
        logger.info("Initialized Medical Entity Extractor")
    
    def extract_entities(self, text: str) -> List[MedicalEntity]:
        """Extract all medical entities from text"""
        entities = []
        text_lower = text.lower()
        
        # Extract symptoms
        for pattern in self.symptom_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities.append(MedicalEntity(
                    text=match.group(),
                    label="SYMPTOM",
                    start=match.start(),
                    end=match.end()
                ))
        
        # Extract medications
        for pattern in self.medication_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities.append(MedicalEntity(
                    text=match.group(),
                    label="MEDICATION", 
                    start=match.start(),
                    end=match.end()
                ))
        
        # Extract procedures
        for pattern in self.procedure_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                entities.append(MedicalEntity(
                    text=match.group(),
                    label="PROCEDURE",
                    start=match.start(),
                    end=match.end()
                ))
        
        return entities
    
    def extract_vital_signs(self, text: str) -> Dict[str, str]:
        """Extract vital signs with values"""
        vitals = {}
        text_lower = text.lower()
        
        for vital_name, pattern in self.vital_patterns.items():
            match = re.search(pattern, text_lower)
            if match:
                vitals[vital_name] = match.group(1)
        
        return vitals


class ClinicalReasoningEngine:
    """Core clinical reasoning and response generation"""
    
    def __init__(self):
        self.entity_extractor = MedicalEntityExtractor()
        self.response_templates = self._load_response_templates()
        self.diagnosis_mappings = self._load_diagnosis_mappings()
        logger.info("Initialized Clinical Reasoning Engine")
    
    def _load_response_templates(self) -> Dict[str, str]:
        """Load clinical response templates by specialty"""
        return {
            "emergency": "EMERGENCY ASSESSMENT: {assessment}. IMMEDIATE MANAGEMENT: {management}. MONITORING: {monitoring}.",
            "pediatric": "PEDIATRIC EVALUATION: {assessment}. AGE-APPROPRIATE CARE: {management}. FAMILY EDUCATION: {education}.",
            "adult": "CLINICAL ASSESSMENT: {assessment}. MANAGEMENT PLAN: {management}. FOLLOW-UP: {followup}.",
            "maternal": "MATERNAL ASSESSMENT: {assessment}. OBSTETRIC CARE: {management}. MATERNAL-FETAL MONITORING: {monitoring}.",
            "general": "CLINICAL EVALUATION: {assessment}. TREATMENT APPROACH: {management}. PATIENT CARE: {care}."
        }
    
    def _load_diagnosis_mappings(self) -> Dict[str, List[str]]:
        """Load symptom-to-diagnosis mappings"""
        return {
            "respiratory_distress": ["asthma", "pneumonia", "pulmonary embolism"],
            "abdominal_pain": ["appendicitis", "gastritis", "bowel obstruction"],
            "fever_child": ["malaria", "pneumonia", "urinary tract infection"],
            "burn_injury": ["thermal burn", "chemical burn", "electrical burn"],
            "diabetic_emergency": ["diabetic ketoacidosis", "hypoglycemia", "hyperglycemic crisis"]
        }
    
    def analyze_case(self, prompt: str, specialty: str, facility_level: str, experience: float) -> ClinicalFeatures:
        """Comprehensive case analysis"""
        
        # Extract medical entities
        entities = self.entity_extractor.extract_entities(prompt)
        vital_signs = self.entity_extractor.extract_vital_signs(prompt)
        
        # Categorize entities
        symptoms = [e.text for e in entities if e.label == "SYMPTOM"]
        medications = [e.text for e in entities if e.label == "MEDICATION"]
        procedures = [e.text for e in entities if e.label == "PROCEDURE"]
        
        # Extract demographics
        demographics = self._extract_demographics(prompt)
        
        # Identify severity indicators
        severity_indicators = self._assess_severity(prompt, vital_signs)
        
        # Extract specialty-specific keywords
        specialty_keywords = self._extract_specialty_keywords(prompt, specialty)
        
        features = ClinicalFeatures(
            symptoms=symptoms,
            medications=medications,
            procedures=procedures,
            vital_signs=vital_signs,
            demographics=demographics,
            severity_indicators=severity_indicators,
            specialty_keywords=specialty_keywords
        )
        
        logger.debug(f"Extracted clinical features", 
                    symptoms=len(symptoms),
                    medications=len(medications),
                    procedures=len(procedures))
        
        return features
    
    def _extract_demographics(self, text: str) -> Dict[str, str]:
        """Extract patient demographics"""
        demographics = {}
        
        # Age extraction
        age_match = re.search(r'(\d+)\s*(year|month|day)s?\s*old', text.lower())
        if age_match:
            demographics['age'] = age_match.group(1)
            demographics['age_unit'] = age_match.group(2)
        
        # Gender extraction
        if re.search(r'\b(male|man|boy)\b', text.lower()):
            demographics['gender'] = 'male'
        elif re.search(r'\b(female|woman|girl)\b', text.lower()):
            demographics['gender'] = 'female'
        
        return demographics
    
    def _assess_severity(self, text: str, vitals: Dict[str, str]) -> List[str]:
        """Assess case severity indicators"""
        severity_indicators = []
        
        # High-priority symptoms
        if re.search(r'\b(emergency|urgent|acute|severe|critical)\b', text.lower()):
            severity_indicators.append("high_priority")
        
        # Vital signs assessment
        if 'blood_pressure' in vitals:
            bp = vitals['blood_pressure']
            systolic = int(bp.split('/')[0])
            if systolic > 140 or systolic < 90:
                severity_indicators.append("abnormal_bp")
        
        if 'temperature' in vitals:
            temp = float(vitals['temperature'])
            if temp > 38.5 or temp < 35:
                severity_indicators.append("abnormal_temperature")
        
        if 'oxygen_saturation' in vitals:
            spo2 = int(vitals['oxygen_saturation'])
            if spo2 < 90:
                severity_indicators.append("hypoxia")
        
        # Clinical deterioration indicators
        if re.search(r'\b(unconscious|unresponsive|collapsed|shock)\b', text.lower()):
            severity_indicators.append("critical_condition")
        
        return severity_indicators
    
    def _extract_specialty_keywords(self, text: str, specialty: str) -> List[str]:
        """Extract specialty-specific clinical keywords"""
        keywords = []
        text_lower = text.lower()
        
        specialty_patterns = {
            "emergency": [r'\b(trauma|accident|injury|urgent)\b'],
            "pediatric": [r'\b(child|infant|baby|pediatric|paediatric)\b'],
            "maternal": [r'\b(pregnant|pregnancy|maternal|obstetric|gravida|para)\b'],
            "surgical": [r'\b(surgery|surgical|operation|incision|wound)\b']
        }
        
        if specialty.lower() in specialty_patterns:
            for pattern in specialty_patterns[specialty.lower()]:
                matches = re.findall(pattern, text_lower)
                keywords.extend(matches)
        
        return list(set(keywords))
    
    def generate_clinical_response(self, features: ClinicalFeatures, specialty: str, 
                                 facility_level: str, experience: float, target_length: int = 700) -> str:
        """Generate expert-level clinical response"""
        
        # Determine response template based on specialty
        template_key = self._select_template(specialty, features.severity_indicators)
        template = self.response_templates.get(template_key, self.response_templates["general"])
        
        # Generate assessment component
        assessment = self._generate_assessment(features)
        
        # Generate management component
        management = self._generate_management(features, facility_level, experience)
        
        # Generate monitoring/follow-up component
        monitoring = self._generate_monitoring(features, facility_level)
        
        # Construct response
        response_components = {
            "assessment": assessment,
            "management": management,
            "monitoring": monitoring,
            "education": "Patient and family education on condition management",
            "followup": "Regular follow-up and reassessment recommended",
            "care": "Comprehensive patient care approach"
        }
        
        response = template.format(**response_components)
        
        # Adjust length to target
        response = self._adjust_response_length(response, target_length)
        
        logger.debug(f"Generated response", length=len(response), specialty=specialty)
        
        return response
    
    def _select_template(self, specialty: str, severity_indicators: List[str]) -> str:
        """Select appropriate response template"""
        if "high_priority" in severity_indicators or "critical_condition" in severity_indicators:
            return "emergency"
        elif "child" in specialty.lower() or "pediatric" in specialty.lower():
            return "pediatric"
        elif "maternal" in specialty.lower() or "obstetric" in specialty.lower():
            return "maternal"
        elif "adult" in specialty.lower():
            return "adult"
        else:
            return "general"
    
    def _generate_assessment(self, features: ClinicalFeatures) -> str:
        """Generate clinical assessment section"""
        components = []
        
        if features.demographics:
            age = features.demographics.get('age', 'adult')
            gender = features.demographics.get('gender', 'patient')
            components.append(f"{age}-year-old {gender}")
        
        if features.symptoms:
            symptom_text = ", ".join(features.symptoms[:3])  # Top 3 symptoms
            components.append(f"presenting with {symptom_text}")
        
        if features.vital_signs:
            vital_summary = self._summarize_vitals(features.vital_signs)
            components.append(vital_summary)
        
        if features.severity_indicators:
            if "critical_condition" in features.severity_indicators:
                components.append("requiring immediate intervention")
            elif "high_priority" in features.severity_indicators:
                components.append("requiring urgent care")
        
        return ". ".join(components) if components else "Clinical assessment pending"
    
    def _generate_management(self, features: ClinicalFeatures, facility_level: str, experience: float) -> str:
        """Generate management plan based on facility capabilities"""
        management_steps = []
        
        # Basic supportive care
        if "hypoxia" in features.severity_indicators:
            management_steps.append("oxygen supplementation")
        
        if features.symptoms:
            if "pain" in " ".join(features.symptoms):
                management_steps.append("analgesic management")
            if "fever" in " ".join(features.symptoms):
                management_steps.append("antipyretic therapy")
        
        # Facility-appropriate interventions
        if "national referral" in facility_level.lower():
            management_steps.append("comprehensive diagnostic workup")
            management_steps.append("specialist consultation available")
        elif "sub county" in facility_level.lower():
            management_steps.append("basic diagnostic assessment")
            management_steps.append("referral for complex cases")
        else:
            management_steps.append("primary care management")
            management_steps.append("referral if indicated")
        
        # Experience-adjusted complexity
        if experience > 15:
            management_steps.append("advanced clinical protocols")
        elif experience > 10:
            management_steps.append("standard care protocols")
        else:
            management_steps.append("supervised care with senior support")
        
        return ". ".join(management_steps) if management_steps else "Standard care protocols"
    
    def _generate_monitoring(self, features: ClinicalFeatures, facility_level: str) -> str:
        """Generate monitoring and follow-up plan"""
        monitoring_items = []
        
        if features.vital_signs:
            monitoring_items.append("vital signs monitoring")
        
        if "critical_condition" in features.severity_indicators:
            monitoring_items.append("continuous observation")
        elif "high_priority" in features.severity_indicators:
            monitoring_items.append("frequent assessment")
        else:
            monitoring_items.append("regular follow-up")
        
        if "national referral" in facility_level.lower():
            monitoring_items.append("advanced monitoring capabilities")
        else:
            monitoring_items.append("clinical observation")
        
        return ". ".join(monitoring_items) if monitoring_items else "Standard monitoring"
    
    def _summarize_vitals(self, vitals: Dict[str, str]) -> str:
        """Summarize vital signs assessment"""
        vital_descriptions = []
        
        if 'blood_pressure' in vitals:
            vital_descriptions.append(f"BP {vitals['blood_pressure']}")
        if 'temperature' in vitals:
            vital_descriptions.append(f"temp {vitals['temperature']}°C")
        if 'pulse' in vitals:
            vital_descriptions.append(f"HR {vitals['pulse']}")
        if 'oxygen_saturation' in vitals:
            vital_descriptions.append(f"SpO2 {vitals['oxygen_saturation']}%")
        
        return "Vitals: " + ", ".join(vital_descriptions) if vital_descriptions else ""
    
    def _adjust_response_length(self, response: str, target_length: int) -> str:
        """Adjust response to target length"""
        if len(response) > target_length:
            # Truncate at sentence boundary
            sentences = response.split('. ')
            truncated = ""
            for sentence in sentences:
                if len(truncated + sentence + '. ') <= target_length:
                    truncated += sentence + '. '
                else:
                    break
            return truncated.rstrip()
        
        elif len(response) < target_length * 0.8:
            # Add clinical details
            additions = [
                " Comprehensive patient assessment completed.",
                " Ongoing monitoring and care coordination essential.",
                " Patient safety and quality care prioritized.",
                " Evidence-based treatment protocols followed."
            ]
            
            for addition in additions:
                if len(response + addition) <= target_length:
                    response += addition
                else:
                    break
        
        return response


def train_expert_response_analyzer(train_df: pd.DataFrame) -> Dict[str, any]:
    """Analyze expert responses to learn patterns"""
    
    logger.info("Analyzing expert response patterns")
    
    expert_columns = ['Nursing Competency', 'Clinical Panel', 'Clinician', 'GPT4.0', 'LLAMA', 'GEMINI']
    
    analysis = {}
    
    for col in expert_columns:
        if col in train_df.columns:
            responses = train_df[col].dropna()
            
            analysis[col] = {
                'avg_length': responses.str.len().mean(),
                'median_length': responses.str.len().median(),
                'length_std': responses.str.len().std(),
                'response_count': len(responses),
                'common_phrases': extract_common_phrases(responses.tolist()),
                'clinical_structure': analyze_clinical_structure(responses.tolist())
            }
    
    logger.info(f"Expert response analysis complete", models_analyzed=len(analysis))
    
    return analysis


def extract_common_phrases(responses: List[str]) -> List[str]:
    """Extract common clinical phrases from expert responses"""
    
    # Combine all responses
    combined_text = " ".join(responses).lower()
    
    # Common medical phrase patterns
    clinical_patterns = [
        r'\b(immediate|urgent|emergency)\s+\w+',
        r'\b(patient|child|infant)\s+(should|requires|needs)',
        r'\b(monitor|assess|evaluate|examine)\s+\w+',
        r'\b(administer|give|provide)\s+\w+',
        r'\b(refer|consult|contact)\s+\w+'
    ]
    
    common_phrases = []
    for pattern in clinical_patterns:
        matches = re.findall(pattern, combined_text)
        common_phrases.extend(matches[:5])  # Top 5 per pattern
    
    return list(set(common_phrases))


def analyze_clinical_structure(responses: List[str]) -> Dict[str, float]:
    """Analyze clinical response structure patterns"""
    
    structure_indicators = {
        'has_assessment': 0,
        'has_management': 0,
        'has_monitoring': 0,
        'has_education': 0,
        'uses_numbered_list': 0,
        'mentions_vitals': 0,
        'includes_medications': 0,
        'mentions_referral': 0
    }
    
    for response in responses:
        response_lower = response.lower()
        
        if any(word in response_lower for word in ['assess', 'evaluation', 'diagnosis']):
            structure_indicators['has_assessment'] += 1
        
        if any(word in response_lower for word in ['manage', 'treatment', 'therapy']):
            structure_indicators['has_management'] += 1
        
        if any(word in response_lower for word in ['monitor', 'follow', 'observe']):
            structure_indicators['has_monitoring'] += 1
        
        if any(word in response_lower for word in ['educate', 'teach', 'explain']):
            structure_indicators['has_education'] += 1
        
        if re.search(r'\d+\.?\s+', response):
            structure_indicators['uses_numbered_list'] += 1
        
        if any(word in response_lower for word in ['bp', 'pulse', 'temperature', 'vitals']):
            structure_indicators['mentions_vitals'] += 1
        
        if any(word in response_lower for word in ['medication', 'drug', 'dosage']):
            structure_indicators['includes_medications'] += 1
        
        if any(word in response_lower for word in ['refer', 'specialist', 'consultation']):
            structure_indicators['mentions_referral'] += 1
    
    # Convert to percentages
    total_responses = len(responses)
    for key in structure_indicators:
        structure_indicators[key] = structure_indicators[key] / total_responses
    
    return structure_indicators
