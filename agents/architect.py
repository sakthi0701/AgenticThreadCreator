
from agno.agent import Agent
import sys
import os
import dspy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_loader import ModelLoader

class ArchitectAgent(Agent):
    """
    An agent that acts as the 'Chief Architect' of the post.
    It takes raw input + research and outputs a Refined Topic, Template, and Detailed Outline.
    """
    
    def __init__(self):
        super().__init__(
            name="ArchitectAgent",
            description="Plans the proper structure, angle, and outline for the thread.",
        )
        self.agentic_model = ModelLoader.get_model()

    def act(self, raw_topic: str, research_context: str):
        """Refines topic and creates outline in one go."""
        print(f"🏗️ ArchitectAgent is designing the blueprint...")
        
        try:
            # Call the consolidated module
            result = self.agentic_model.architect_module(
                raw_topic=raw_topic,
                research_context=research_context
            )
            
            print(f"   -> Refined Topic: {result.refined_topic}")
            print(f"   -> Template: {result.template_type}")
            
            # The outline is returned as a list of strings usually (via dspy output)
            # but we should ensure it's a list.
            outline = result.thread_outline
            if isinstance(outline, str):
                # Basic parsing if dspy returns a string representation
                import ast
                try:
                    outline = ast.literal_eval(outline)
                except:
                    # Fallback split
                    outline = [x.strip() for x in outline.split('\n') if x.strip()]
            
            return {
                "refined_topic": result.refined_topic,
                "template": result.template_type,
                "outline": outline,
                "psychological_trigger": getattr(result, "psychological_trigger", "Social Proof")
            }
            
        except Exception as e:
            print(f"⚠️ ArchitectAgent failed: {e}")
            # Fallback
            return {
                "refined_topic": raw_topic,
                "template": "General", 
                "outline": [raw_topic]
            }
