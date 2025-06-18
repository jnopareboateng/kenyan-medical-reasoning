"""
Model Cache Management Utility
Provides centralized cache control for all clinical models
"""

import torch
import gc
from pathlib import Path
from utils.logger import CompetitionLogger

logger = CompetitionLogger("ModelCacheManager")

class ModelCacheManager:
    """Centralized cache management for all clinical models"""
    
    @staticmethod
    def cleanup_all_models():
        """Clean up ALL cached models from memory across all model types"""
        
        # Import here to avoid circular imports
        try:
            from core.phi4_model import ClinicalPhi4Model
            ClinicalPhi4Model.clear_cache()
            logger.info("✅ Phi-4 cache cleared")
        except ImportError:
            logger.warning("Phi-4 model not available")
        
        try:
            from core.meditron_model import ClinicalMeditronModel
            ClinicalMeditronModel.clear_cache()
            logger.info("✅ Meditron cache cleared")
        except ImportError:
            logger.warning("Meditron model not available")
        
        try:
            from core.llama32_model import ClinicalLlama32Model
            ClinicalLlama32Model.clear_cache()
            logger.info("✅ Llama-3.2 cache cleared")
        except ImportError:
            logger.warning("Llama-3.2 model not available")
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        logger.info("🧹 ALL model caches cleared and memory freed")
    
    @staticmethod
    def get_cache_info():
        """Get information about current cache usage"""
        
        cache_info = {
            "phi4_cached_models": 0,
            "meditron_cached_models": 0,
            "llama32_cached_models": 0,
            "total_cached_models": 0,
            "gpu_memory_allocated": 0.0,
            "gpu_memory_reserved": 0.0
        }
        
        # Check each model's cache
        try:
            from core.phi4_model import ClinicalPhi4Model
            cache_info["phi4_cached_models"] = len(ClinicalPhi4Model._model_cache)
        except ImportError:
            pass
        
        try:
            from core.meditron_model import ClinicalMeditronModel
            cache_info["meditron_cached_models"] = len(ClinicalMeditronModel._model_cache)
        except ImportError:
            pass
        
        try:
            from core.llama32_model import ClinicalLlama32Model
            cache_info["llama32_cached_models"] = len(ClinicalLlama32Model._model_cache)
        except ImportError:
            pass
        
        cache_info["total_cached_models"] = (
            cache_info["phi4_cached_models"] + 
            cache_info["meditron_cached_models"] + 
            cache_info["llama32_cached_models"]
        )
        
        # GPU memory info
        if torch.cuda.is_available():
            cache_info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / 1e9
            cache_info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / 1e9
        
        return cache_info
    
    @staticmethod
    def print_cache_status():
        """Print detailed cache status for debugging"""
        
        info = ModelCacheManager.get_cache_info()
        
        print("🔍 MODEL CACHE STATUS:")
        print(f"  Phi-4 models cached: {info['phi4_cached_models']}")
        print(f"  Meditron models cached: {info['meditron_cached_models']}")
        print(f"  Llama-3.2 models cached: {info['llama32_cached_models']}")
        print(f"  Total models in memory: {info['total_cached_models']}")
        
        if torch.cuda.is_available():
            print(f"  GPU Memory Allocated: {info['gpu_memory_allocated']:.2f}GB")
            print(f"  GPU Memory Reserved: {info['gpu_memory_reserved']:.2f}GB")
        else:
            print("  GPU: Not available")
        
        if info['total_cached_models'] == 0:
            print("  ✅ No models cached - memory is clean")
        else:
            print(f"  ⚠️ {info['total_cached_models']} models using memory")
    
    @staticmethod
    def emergency_cleanup():
        """Nuclear option: clear everything and force garbage collection"""
        
        logger.warning("🚨 EMERGENCY CLEANUP: Clearing all caches and forcing GC")
        
        # Clear all model caches
        ModelCacheManager.cleanup_all_models()
        
        # Force multiple garbage collection passes
        for i in range(3):
            collected = gc.collect()
            logger.info(f"GC pass {i+1}: collected {collected} objects")
        
        # Clear CUDA cache multiple times
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.synchronize()
        
        logger.info("💀 Emergency cleanup completed")

# Convenience functions for notebook usage
def cleanup_all():
    """Quick cleanup function for notebook cells"""
    ModelCacheManager.cleanup_all_models()

def cache_status():
    """Quick status check for notebook cells"""
    ModelCacheManager.print_cache_status()

def emergency():
    """Emergency cleanup for notebook cells"""
    ModelCacheManager.emergency_cleanup()
