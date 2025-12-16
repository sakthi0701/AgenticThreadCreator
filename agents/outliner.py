# agents/outliner.py

from agno.agent import Agent
import sys
import os
# import dspy # Not strictly needed if using Loader
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_loader import ModelLoader
from utils.parsing import safe_parse_list
from dspy_module import configure_llm

class OutlinerAgent(Agent):
    """
    An agent that generates a thread outline from a topic, hook,
    research context, and a mandatory template.
    """
    
    def __init__(self):
        super().__init__(
            name="OutlinerAgent",
            description="Generates a bulleted outline from a hook and research.",
        )
        self.agentic_model = ModelLoader.get_model()

    # --- UPDATED THE 'act' SIGNATURE ---
    def act(self, topic: str, hook: str, research_context: str, template_type: str): 
        """
        Generates a list of outline bullet points, strictly
        following the template_type.
        """
        print(f"📑 OutlinerAgent creating outline for '{topic}' (Template: {template_type})...")
        
        # --- UPDATED THE MODULE CALL ---
        result = self.agentic_model.outline_module(
            topic=topic, 
            hook=hook, 
            research_context=research_context,
            template_type=template_type  # <-- PASS IT TO THE MODULE
        )
        
        # --- ROBUST PARSING LOGIC (Unchanged) ---
        raw_outline = result.outline
        outline_list = []
        
        # --- ROBUST PARSING LOGIC ---
        raw_outline = result.outline
        outline_list = safe_parse_list(raw_outline, fix_with_llm=True)
        
        if not outline_list:
            print("⚠️ Error: Outline parsing failed completely.")
            outline_list = [raw_outline]
            
        print("✅ OutlinerAgent finished.")
        return {"outline": outline_list}