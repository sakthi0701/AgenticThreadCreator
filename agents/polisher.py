
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
        if isinstance(draft_content, list) and len(draft_content) == 1 and len(draft_content[0]) > 500:
             print("⚠️ Detected Wall of Text. Attempting pre-split by paragraphs.")
             # Split by double newlines to simulate distinct tweets
             draft_content = [p for p in draft_content[0].split('\n\n') if p.strip()]

        try:
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

             # --- SAFETY FILTER FOR LINKEDIN ---
             if platform == "LinkedIn":
                 import re
                 def clean_linkedin_text(text):
                     if not isinstance(text, str): return text
                     # 1. Normalize bullets: "* item" -> "• item" (Aesthetic fix)
                     text = re.sub(r'^\s*\*\s+', '• ', text, flags=re.MULTILINE)
                     
                     # 2. Remove Markdown artifacts
                     text = text.replace('**', '') # Remove bolding
                     text = text.replace('##', '') # Remove headers
                     
                     # 3. Cleanup remaining * (likely italics) but keep text
                     # e.g. *italic* -> italic
                     text = text.replace('*', '')
                     
                     return text.strip()

                 result.refined_hook = clean_linkedin_text(result.refined_hook)
                 result.refined_cta = clean_linkedin_text(result.refined_cta)
                 
                 if isinstance(refined_body, list):
                     refined_body = [clean_linkedin_text(t) for t in refined_body]
                 else:
                     refined_body = clean_linkedin_text(refined_body)

             return {
                 "refined_hook": result.refined_hook,
                 "refined_tweets": refined_body,
                 "refined_cta": result.refined_cta
             }
        except Exception as e:
             print(f"⚠️ PolisherAgent Full Refine failed: {e}")
             return {"error": str(e)}
