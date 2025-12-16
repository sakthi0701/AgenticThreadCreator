
from agno.agent import Agent
import sys
import os
import dspy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_loader import ModelLoader
from dspy_module import configure_llm

class RefinerAgent(Agent):
    """
    An agent that transforms a raw 'Brain Dump' into a structured Topic and Angle.
    This acts as the translation layer between the user's stream-of-consciousness and the Writer.
    """
    
    def __init__(self):
        super().__init__(
            name="RefinerAgent",
            description="Refines raw input into a clear topic and angle.",
        )
        self.agentic_model = ModelLoader.get_model()

    def act(self, raw_input: str, research_context: str = ""):
        """Refines the raw input using research context."""
        print(f"💎 RefinerAgent is polishing the raw input...")
        
        try:
            # Call the module from the loaded model
            result = self.agentic_model.refiner_module(
                raw_input=raw_input,
                research_context=research_context
            )
            
            print(f"   -> Refined Topic: {result.refined_topic}")
            print(f"   -> Detected Angle: {result.angle}")
            
            return {
                "refined_topic": result.refined_topic,
                "angle": result.angle,
                "key_points": result.key_points
            }
            
        except Exception as e:
            print(f"⚠️ RefinerAgent failed: {e}")
            # Fallback: Just use the first sentence as topic
            return {
                "refined_topic": raw_input[:100] + "...",
                "angle": "General", 
                "key_points": raw_input
            }
