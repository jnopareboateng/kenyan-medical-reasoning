"""
Data Processing Core Module
Production-grade data handling for medical AI competition
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import re
from dataclasses import dataclass
import json

from utils.logger import get_logger
from utils.paths import PATHS

logger = get_logger("data_processing")


@dataclass
class MedicalCase:
    """Structured representation of a medical case"""
    case_id: str
    county: str
    health_level: str
    experience_years: Optional[float]
    prompt: str
    specialty: str
    panel: str
    target_response: Optional[str] = None
    snomed_code: Optional[str] = None
    
    def __post_init__(self):
        """Extract medical features after initialization"""
        self.prompt_length = len(self.prompt)
        self.complexity_level = self._assess_complexity()
        self.extracted_entities = self._extract_medical_entities()
    
    def _assess_complexity(self) -> str:
        """Assess case complexity based on prompt length and content"""
        if self.prompt_length < 500:
            return "simple"
        elif self.prompt_length < 1000:
            return "medium" 
        elif self.prompt_length < 2000:
            return "complex"
        else:
            return "very_complex"
    
    def _extract_medical_entities(self) -> Dict[str, List[str]]:
        """Extract medical entities from prompt text"""
        entities = {
            "symptoms": [],
            "medications": [],
            "procedures": [],
            "measurements": []
        }
        
        # Basic regex patterns for medical entities
        patterns = {
            "symptoms": r'\b(pain|fever|headache|nausea|vomiting|bleeding|swelling|difficulty|weakness)\b',
            "medications": r'\b(insulin|antibiotics|analgesics|paracetamol|ibuprofen|morphine)\b',
            "procedures": r'\b(x-ray|ct scan|ultrasound|biopsy|surgery|intubation)\b',
            "measurements": r'\b(\d+\.?\d*\s*(mg|ml|kg|bpm|mmhg|°c|%))\b'
        }
        
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, self.prompt.lower())
            entities[entity_type] = list(set(matches))
        
        return entities


class MedicalDataProcessor:
    """Core data processing for medical cases"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.medical_cases: List[MedicalCase] = []
        logger.info("Initialized MedicalDataProcessor")
    
    def load_training_data(self, file_path: Path) -> List[MedicalCase]:
        """Load and process training data"""
        logger.info(f"Loading training data from {file_path}")
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Raw training data loaded: {len(df)} cases, {len(df.columns)} columns")
            
            # Convert to structured medical cases
            cases = []
            for _, row in df.iterrows():
                case = MedicalCase(
                    case_id=row['Master_Index'],
                    county=row['County'],
                    health_level=row['Health level'],
                    experience_years=row.get('Years of Experience'),
                    prompt=row['Prompt'],
                    specialty=row['Nursing Competency'],
                    panel=row['Clinical Panel'],
                    target_response=row.get('Clinician'),
                    snomed_code=row.get('DDX SNOMED')
                )
                cases.append(case)
            
            self.medical_cases = cases
            logger.info(f"Processed {len(cases)} medical cases")
            
            # Log processing statistics
            self._log_data_statistics(cases)
            
            return cases
            
        except Exception as e:
            logger.error(f"Failed to load training data", exception=e)
            raise
    
    def load_test_data(self, file_path: Path) -> List[MedicalCase]:
        """Load test data for prediction"""
        logger.info(f"Loading test data from {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            
            cases = []
            for _, row in df.iterrows():
                case = MedicalCase(
                    case_id=row['Master_Index'],
                    county=row['County'],
                    health_level=row['Health level'],
                    experience_years=row.get('Years of Experience'),
                    prompt=row['Prompt'],
                    specialty=row['Nursing Competency'],
                    panel=row['Clinical Panel']
                )
                cases.append(case)
            
            logger.info(f"Loaded {len(cases)} test cases")
            return cases
            
        except Exception as e:
            logger.error(f"Failed to load test data", exception=e)
            raise
    
    def _log_data_statistics(self, cases: List[MedicalCase]) -> None:
        """Log comprehensive data statistics"""
        
        # Basic statistics
        total_cases = len(cases)
        counties = [case.county for case in cases]
        specialties = [case.specialty for case in cases]
        complexity_levels = [case.complexity_level for case in cases]
        
        stats = {
            "total_cases": total_cases,
            "unique_counties": len(set(counties)),
            "county_distribution": dict(pd.Series(counties).value_counts()),
            "unique_specialties": len(set(specialties)),
            "specialty_distribution": dict(pd.Series(specialties).value_counts().head(10)),
            "complexity_distribution": dict(pd.Series(complexity_levels).value_counts()),
            "avg_prompt_length": np.mean([case.prompt_length for case in cases]),
            "experience_stats": {
                "min": min([c.experience_years for c in cases if c.experience_years is not None]),
                "max": max([c.experience_years for c in cases if c.experience_years is not None]),
                "mean": np.mean([c.experience_years for c in cases if c.experience_years is not None])
            }
        }
        
        logger.info(f"Processed training data: {stats}")
    
    def create_train_validation_split(self, test_size: float = 0.2, random_state: int = 42) -> Tuple[List[MedicalCase], List[MedicalCase]]:
        """Create stratified train/validation split by medical specialty"""
        
        from sklearn.model_selection import train_test_split
        
        # Group by specialty for stratification
        specialties = [case.specialty for case in self.medical_cases]
        
        train_indices, val_indices = train_test_split(
            range(len(self.medical_cases)),
            test_size=test_size,
            random_state=random_state,
            stratify=specialties
        )
        
        train_cases = [self.medical_cases[i] for i in train_indices]
        val_cases = [self.medical_cases[i] for i in val_indices]
        
        logger.info(f"Created train/validation split: {len(train_cases)} train, {len(val_cases)} validation")
        
        return train_cases, val_cases
    
    def get_feature_summary(self) -> Dict[str, Any]:
        """Get comprehensive feature summary for model training"""
        
        if not self.medical_cases:
            logger.warning("No medical cases loaded")
            return {}
        
        # Extract all unique values for categorical features
        counties = set(case.county for case in self.medical_cases)
        health_levels = set(case.health_level for case in self.medical_cases)
        specialties = set(case.specialty for case in self.medical_cases)
        
        # Calculate feature statistics
        prompt_lengths = [case.prompt_length for case in self.medical_cases]
        
        summary = {
            "categorical_features": {
                "counties": sorted(list(counties)),
                "health_levels": sorted(list(health_levels)),
                "specialties": sorted(list(specialties))
            },
            "numerical_features": {
                "prompt_length": {
                    "min": min(prompt_lengths),
                    "max": max(prompt_lengths),
                    "mean": np.mean(prompt_lengths),
                    "std": np.std(prompt_lengths)
                }
            },
            "total_cases": len(self.medical_cases)
        }
        
        logger.info(f"Feature summary: {summary}")
        return summary


def quick_data_analysis(data_path: Path) -> Dict[str, Any]:
    """Quick analysis for immediate insights"""
    
    logger.info("🔥 STARTING QUICK DATA ANALYSIS")
    
    # Load raw data
    df = pd.read_csv(data_path)
    
    analysis = {
        "dataset_shape": df.shape,
        "columns": list(df.columns),
        "missing_data": df.isnull().sum().to_dict(),
        "response_lengths": {
            col: df[col].str.len().describe().to_dict() 
            for col in ['Nursing Competency', 'Clinical Panel', 'Clinician'] 
            if col in df.columns
        },
        "specialties": df['Nursing Competency'].value_counts().head(10).to_dict(),
        "counties": df['County'].value_counts().to_dict(),
        "facility_levels": df['Health level'].value_counts().to_dict()
    }
    
    logger.info(f"Quick analysis complete: {analysis}")
    
    # Critical insights
    logger.info("📊 CRITICAL INSIGHTS:")
    logger.info(f"- Total cases: {df.shape[0]}")
    logger.info(f"- Unique specialties: {df['Nursing Competency'].nunique()}")
    logger.info(f"- Avg clinician response length: {df['Clinician'].str.len().mean():.0f} chars")
    logger.info(f"- Most common specialty: {df['Nursing Competency'].mode()[0]}")
    
    return analysis
