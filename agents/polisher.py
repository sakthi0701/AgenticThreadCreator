
from agno.agent import Agent
import sys
import os
import dspy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_loader import ModelLoader
from utils.parsing import safe_parse_list

class PolisherAgent(Agent):
    """
    An agent that Polishes the draft (Editing + TLDR + CTA).
    """
    
    def __init__(self):
        super().__init__(
            name="PolisherAgent",
            description="Edits the thread and adds TLDR/CTA.",
        )
        self.agentic_model = ModelLoader.get_model()

    def act(self, draft_content: list, psychological_trigger: str, topic: str = ""):
        """Edits thread and generates ancillary content."""
        print(f"✨ PolisherAgent is applying the final shine...")
        
        try:
            # Convert list to string for input
            draft_str = str(draft_content)
            
            result = self.agentic_model.polisher_module(draft_content=draft_str, psychological_trigger=psychological_trigger)
            
            # Parse the polished list
            polished_list = safe_parse_list(result.polished_content, fix_with_llm=True)
            if not polished_list:
                polished_list = [result.polished_content]
                
            return {
                "edited_tweets": polished_list,
                "tldr": result.tldr,
                "cta": result.cta
            }
            
        except Exception as e:
            print(f"⚠️ PolisherAgent failed: {e}")
            return {
                "edited_tweets": draft_content,
                "tldr": "Error generating TLDR", 
                "cta": "Error generating CTA"
            }

    def refine_full_post(self, hook, tweets, cta, prompt, platform="Twitter"):
        """Refines the entire post based on user instruction."""
        print(f"✨ PolisherAgent is refining the FULL post (Platform: {platform})...")
        try:
             # Call the module
             result = self.agentic_model.refine_full_module(
                 current_hook=hook,
                 current_body=str(tweets),
                 current_cta=cta,
                 instruction=prompt,
                 platform=platform
             )
             
             # Parse body
             refined_body = safe_parse_list(result.refined_body, fix_with_llm=True)
             if not refined_body:
                 refined_body = [result.refined_body]

             return {
                 "refined_hook": result.refined_hook,
                 "refined_tweets": refined_body,
                 "refined_cta": result.refined_cta
             }
        except Exception as e:
             print(f"⚠️ PolisherAgent Full Refine failed: {e}")
             return {"error": str(e)}
