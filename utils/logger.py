"""
Centralized Logging System
Production-grade logging for medical AI competition
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime
import json


class CompetitionLogger:
    """
    Centralized logger for the medical AI competition
    Tracks everything - data processing, model training, evaluation, errors
    """
    
    def __init__(self, name: str, log_dir: Optional[Path] = None, level: str = "INFO"):
        self.name = name
        self.log_dir = log_dir or Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup file and console handlers"""
        
        # File handler - detailed logs
        log_file = self.log_dir / f"{self.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler - important messages only
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(filename)s:%(lineno)d | %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(levelname)s | %(message)s'
        )
        
        file_handler.setFormatter(detailed_formatter)
        console_handler.setFormatter(simple_formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message: str, **kwargs):
        """Log info message with optional structured data"""
        if kwargs:
            message += f" | Data: {json.dumps(kwargs)}"
        self.logger.info(message)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        if kwargs:
            message += f" | Data: {json.dumps(kwargs)}"
        self.logger.warning(message)
    
    def error(self, message: str, exception: Optional[Exception] = None, **kwargs):
        """Log error message with optional exception"""
        if kwargs:
            message += f" | Data: {json.dumps(kwargs)}"
        if exception:
            self.logger.error(message, exc_info=exception)
        else:
            self.logger.error(message)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        if kwargs:
            message += f" | Data: {json.dumps(kwargs)}"
        self.logger.debug(message)
    
    def critical(self, message: str, **kwargs):
        """Log critical message - competition-ending issues"""
        if kwargs:
            message += f" | Data: {json.dumps(kwargs)}"
        self.logger.critical(message)
    
    def competition_phase(self, phase: str, status: str = "START", **metrics):
        """Log competition phase changes with metrics"""
        message = f"🎯 PHASE: {phase} | STATUS: {status}"
        if metrics:
            message += f" | METRICS: {json.dumps(metrics)}"
        self.logger.info(message)
    
    def model_performance(self, model_name: str, metrics: dict):
        """Log model performance metrics"""
        message = f"📊 MODEL: {model_name} | PERFORMANCE: {json.dumps(metrics)}"
        self.logger.info(message)
    
    def data_stats(self, dataset: str, stats: dict):
        """Log dataset statistics"""
        message = f"📈 DATASET: {dataset} | STATS: {json.dumps(stats)}"
        self.logger.info(message)


# Global logger instances
_loggers = {}

def get_logger(name: str, log_dir: Optional[Path] = None) -> CompetitionLogger:
    """Get or create a logger instance"""
    if name not in _loggers:
        _loggers[name] = CompetitionLogger(name, log_dir)
    return _loggers[name]


# Convenience functions
def log_phase(phase: str, status: str = "START", **metrics):
    """Quick phase logging"""
    logger = get_logger("competition")
    logger.competition_phase(phase, status, **metrics)

def log_model(model_name: str, metrics: dict):
    """Quick model performance logging"""
    logger = get_logger("models")
    logger.model_performance(model_name, metrics)

def log_data(dataset: str, stats: dict):
    """Quick data statistics logging"""
    logger = get_logger("data")
    logger.data_stats(dataset, stats)
