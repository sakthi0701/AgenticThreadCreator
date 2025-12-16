# agents/ancillary.py

from agno.agent import Agent
import sys
import os
import dspy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_loader import ModelLoader
from dspy_module import configure_llm

class AncillaryAgent(Agent):
    """
    An agent that generates the TL;DR and CTA for a thread.
    """
    
    def __init__(self):
        super().__init__(
            name="AncillaryAgent",
            description="Generates TL;DR and CTA from a topic and outline.",
        )
        self.agentic_model = ModelLoader.get_model()

    def act(self, topic: str, full_thread_list: list):
        """Generates a tldr and cta."""
        print(f"🎯 AncillaryAgent is generating TL;DR and CTA...")
        
        # Pass the thread list directly
        result = self.agentic_model.ancillary_module(topic=topic, full_thread_list=full_thread_list)
            
        print("✅ AncillaryAgent finished.")
        return {
            "tldr": result.tldr,
            "cta": result.cta
        }