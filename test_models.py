"""Quick test to verify all models load from cache"""
import sys
sys.path.insert(0, 'backend')

from agents.agent_models_loader import MODEL_LOADERS

print("🧪 Testing model loading from cache...\n")

for agent_type, loader_func in MODEL_LOADERS.items():
    try:
        print(f"Testing {agent_type}...", end=" ")
        model = loader_func()
        print("✅ LOADED")
    except Exception as e:
        print(f"❌ FAILED: {e}")

print("\n🎉 Model loading test complete!")


