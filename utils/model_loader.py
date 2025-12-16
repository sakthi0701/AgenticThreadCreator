
import os
import dspy
from warnings import warn
import sys

# Add parent directory to path so we can import dspy_module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dspy_module import AgenticThreadCreator, configure_llm

class ModelLoader:
    """
    Singleton-like loader for the AgenticThreadCreator.
    Ensures we only load the heavy JSON file once.
    """
    _instance = None
    _model = None

    @classmethod
    def get_model(cls):
        """Returns the shared, pre-loaded model instance."""
        if cls._model is None:
            cls._load_model()
        return cls._model

    @classmethod
    def _load_model(cls):
        print("🧠 ModelLoader: Initializing AgenticThreadCreator...")
        
        # 1. Ensure LLM is configured (critical!)
        configure_llm()
        
        # 2. Instantiate the blank model
        model = AgenticThreadCreator()
        
        # 3. Attempt to load compiled weights
        model_path = os.path.join(os.path.dirname(__file__), '..', 'compiled_thread_creator.json')
        model_path = os.path.abspath(model_path)
        
        if os.path.exists(model_path):
            try:
                print(f"📦 ModelLoader: Found compiled model at {model_path}. Loading...")
                model.load(model_path)
                print("✅ ModelLoader: Successfully loaded compiled weights (The Brain is effectively connected!)")
            except Exception as e:
                print(f"⚠️ ModelLoader: Failed to load compiled weights: {e}")
                print("⚠️ ModelLoader: Falling back to uncompiled (Zero-Shot) mode.")
        else:
            print("📝 ModelLoader: No compiled model found. Using uncompiled (Zero-Shot) mode.")
            
        cls._model = model
        return model
