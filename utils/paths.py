"""
Project Configuration Management
Production-grade config handling for medical AI system
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass


@dataclass
class ProjectPaths:
    """Centralized path management"""
    ROOT: Path
    DATA: Path
    CORE: Path
    UTILS: Path
    LOGS: Path
    MODELS: Path
    RESULTS: Path
    CONFIGS: Path

    @classmethod
    def setup(cls, root_path: str) -> 'ProjectPaths':
        """Initialize all project paths"""
        root = Path(root_path)
        paths = cls(
            ROOT=root,
            DATA=root / "data",
            CORE=root / "core",
            UTILS=root / "utils", 
            LOGS=root / "logs",
            MODELS=root / "models",
            RESULTS=root / "results",
            CONFIGS=root / "configs"
        )
        
        # Create directories if they don't exist
        for path in [paths.DATA, paths.MODELS, paths.RESULTS, paths.CONFIGS]:
            path.mkdir(exist_ok=True)
            
        # Additional directories
        (root / "docs").mkdir(exist_ok=True)
        (root / "scripts").mkdir(exist_ok=True)
            
        return paths


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration: {e}")


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """Save configuration to YAML file"""
    try:
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(config, file, default_flow_style=False, indent=2)
    except Exception as e:
        raise ValueError(f"Failed to save configuration: {e}")


def get_env_var(var_name: str, default: Optional[str] = None) -> str:
    """Get environment variable with optional default"""
    value = os.getenv(var_name, default)
    if value is None:
        raise ValueError(f"Environment variable {var_name} not found and no default provided")
    return value


# Initialize project paths (will be set by main application)
PATHS: Optional[ProjectPaths] = None

def init_paths(root_path: str) -> ProjectPaths:
    """Initialize project paths globally"""
    global PATHS
    PATHS = ProjectPaths.setup(root_path)
    return PATHS

def get_project_paths() -> Dict[str, Path]:
    """Get project paths as dictionary for compatibility"""
    if PATHS is None:
        # Auto-detect project root and initialize
        current_file = Path(__file__)
        project_root = current_file.parent.parent
        init_paths(str(project_root))
    
    return {
        'project_root': PATHS.ROOT,
        'data': PATHS.DATA,
        'train_data': PATHS.DATA / 'train.csv',
        'test_data': PATHS.DATA / 'test.csv',
        'core': PATHS.CORE,
        'utils': PATHS.UTILS,
        'logs': PATHS.LOGS,
        'models': PATHS.MODELS,
        'results': PATHS.RESULTS,
        'configs': PATHS.CONFIGS
    }
