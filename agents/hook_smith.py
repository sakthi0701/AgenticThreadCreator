from agno.agent import Agent
import sys
import os
import dspy
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_loader import ModelLoader
from utils.parsing import safe_parse_list
from dspy_module import configure_llm

class HookSmithAgent(Agent):
    """
    An agent that specializes in generating multiple hook options for a topic.
    Uses the compiled SINGLE-HOOK generator multiple times to get high-quality options.
    """
    
    def __init__(self):
        super().__init__(
            name="HookSmithAgent",
            description="Generates 5 compelling hooks for a given topic and template.",
        )
        self.agentic_model = ModelLoader.get_model()

    def act(self, topic: str, angle: str = "Viral/Contrarian"):
        """Generates 5 hooks in a single batch call."""
        print(f" 🎣 HookSmithAgent is casting the net (Batch Mode)...")
        
        try:
            # Single Call!
            # We assume the module now has a 'generate_batch' method or we call the module which uses BatchHookSignature
            # angle arg is mapped to primary_trigger for the new signature
            result = self.agentic_model.hook_module(topic=topic, primary_trigger=angle)
            
            hooks = result.hooks_list
            
            # Ensure it's a list
            if isinstance(hooks, str):
                import ast
                try:
                    hooks = ast.literal_eval(hooks)
                except:
                    # Fallback: assume lines are hooks
                    hooks = [h.strip() for h in hooks.split('\n') if h.strip()]
            
            # Ensure we have 5 items (or at least a list)
            if not isinstance(hooks, list):
                hooks = [str(hooks)]
                
            print(f"   -> Generated {len(hooks)} hooks.")
            for i, h in enumerate(hooks):
                print(f"      {i+1}. {h[:40]}...")

            return {"hooks": hooks}

        except Exception as e:
            print(f"⚠️ HookSmithAgent Batch failed: {e}")
            return {"hooks": ["Error generating hooks via batch mode."]}