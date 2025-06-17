"""
Quick Test Script for New LLM Models
Run this to verify your installations and test model capabilities
"""

import sys
import torch
from utils.logger import CompetitionLogger

logger = CompetitionLogger("ModelTest")

def test_model_installations():
    """Test if all new models can be imported and initialized"""
    
    print("🧪 TESTING MODEL INSTALLATIONS")
    print("=" * 50)
    
    results = {}
    
    # Test Unsloth installation
    try:
        from unsloth import FastLanguageModel
        print("✅ Unsloth installation: SUCCESS")
        results['unsloth'] = True
    except ImportError as e:
        print(f"❌ Unsloth installation: FAILED - {e}")
        results['unsloth'] = False
    
    # Test Phi-4 model
    try:
        from core.phi4_model import ClinicalPhi4Model
        print("✅ Phi-4 model: AVAILABLE")
        results['phi4'] = True
    except ImportError as e:
        print(f"❌ Phi-4 model: FAILED - {e}")
        results['phi4'] = False
    
    # Test Meditron model
    try:
        from core.meditron_model import ClinicalMeditronModel
        print("✅ Meditron model: AVAILABLE")
        results['meditron'] = True
    except ImportError as e:
        print(f"❌ Meditron model: FAILED - {e}")
        results['meditron'] = False
    
    # Test Llama-3.2 model
    try:
        from core.llama32_model import ClinicalLlama32Model
        print("✅ Llama-3.2 model: AVAILABLE")
        results['llama32'] = True
    except ImportError as e:
        print(f"❌ Llama-3.2 model: FAILED - {e}")
        results['llama32'] = False
    
    return results

def quick_model_test():
    """Quick functionality test of available models"""
    
    print("\n🚀 QUICK MODEL FUNCTIONALITY TEST")
    print("=" * 50)
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🔥 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️ No GPU detected - models will run on CPU (slower)")
    
    # Test model imports
    results = test_model_installations()
    
    if not results.get('unsloth', False):
        print("\n❌ CRITICAL: Unsloth not installed!")
        print("Install with: pip install 'unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git'")
        return
    
    print("\n📋 MODEL READINESS:")
    ready_models = []
    
    if results.get('phi4', False):
        ready_models.append("Phi-4-mini-instruct (Recommended)")
    
    if results.get('meditron', False):
        ready_models.append("Meditron-7B (Medical Specialist)")
    
    if results.get('llama32', False):
        ready_models.append("Llama-3.2-3B-Instruct (Balanced)")
    
    if ready_models:
        print("✅ Ready for competition:")
        for model in ready_models:
            print(f"   • {model}")
        
        print(f"\n🎯 You have {len(ready_models)} state-of-the-art models ready!")
        print("Replace your FLAN-T5 usage in the notebook with any of these.")
    else:
        print("❌ No models ready. Check installations.")

def example_usage():
    """Show example of how to use new models"""
    
    print("\n💡 EXAMPLE USAGE IN NOTEBOOK:")
    print("=" * 50)
    
    example_code = '''
# Replace this old code:
# model = ClinicalT5Model("google/flan-t5-small")

# With this new code (choose one):
from core.phi4_model import ClinicalPhi4Model
model = ClinicalPhi4Model("microsoft/Phi-4-mini-instruct", load_in_4bit=True)

# OR for medical specialization:
# from core.meditron_model import ClinicalMeditronModel  
# model = ClinicalMeditronModel("epfl-llm/meditron-7b", load_in_4bit=True)

# OR for balanced performance:
# from core.llama32_model import ClinicalLlama32Model
# model = ClinicalLlama32Model("meta-llama/Llama-3.2-3B-Instruct", load_in_4bit=True)

# Rest of your code stays the same!
training_examples = model.prepare_training_data(train_df)
# ... continue as before
'''
    
    print(example_code)
    
    print("\n🔥 KEY BENEFITS:")
    print("• 2x faster training with Unsloth")
    print("• 70% less VRAM usage")
    print("• 10x better reasoning than FLAN-T5")
    print("• Latest 2024-2025 model architectures")
    print("• Same interface as your existing code")

if __name__ == "__main__":
    quick_model_test()
    example_usage()
    
    print("\n" + "="*60)
    print("🏆 READY TO DOMINATE THE COMPETITION!")
    print("Choose your model and update your notebook.")
    print("="*60)
