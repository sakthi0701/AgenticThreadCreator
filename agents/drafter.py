# agents/drafter.py

from agno.agent import Agent
import sys
import os
import dspy
import ast

# This makes it possible to import from the parent directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_loader import ModelLoader
from utils.parsing import safe_parse_list
from dspy_module import configure_llm

class DrafterAgent(Agent):
    """
    An agent specialized in drafting Twitter threads from an outline.
    """
    
    def __init__(self):
        # Initialize the parent Agent class
        super().__init__(
            name="DrafterAgent",
            description="Drafts a complete Twitter thread from an outline.",
            # No tools needed, it uses a DSPy module
        )
        self.agentic_model = ModelLoader.get_model()


    def act(self, topic: str, outline: list, hooks_list: list, psychological_trigger: str, platform: str = "Twitter"):
        """The main action the agent performs."""
        print(f"🎨 DrafterAgent is now drafting a thread from outline (Platform: {platform})...")
        
        # Call the drafting module directly with the outline
        result = self.agentic_model.drafting_module(
            topic=topic, 
            outline=str(outline), 
            hooks_list=str(hooks_list),
            psychological_trigger=psychological_trigger, 
            platform=platform
        )
        
        # Handle Platform differences
        if platform == "LinkedIn":
             # result has 'post_body' (string) and 'structure_plan'
             print(f"🐛 DEBUG RAW LINKEDIN: {getattr(result, 'post_body', 'Missing')!r}")
             post_body = getattr(result, "post_body", "")
             # Wrap in list to maintain contract with downstream agents
             tweet_list = [post_body]
             selected_hook = "LinkedIn Post" # Logic handled in post body
             
        else:
            # Twitter: result has 'tweet_thread' (list of strings)
            print(f"🐛 DEBUG RAW DRAFT: {result.tweet_thread!r}")
            # Safely parse the output back into a list
            tweet_list = safe_parse_list(result.tweet_thread, fix_with_llm=True)
            if not tweet_list:
                 print("⚠️ Warning: Could not parse drafted thread, returning raw string.")
                 tweet_list = [result.tweet_thread]
            selected_hook = getattr(result, "selected_hook", tweet_list[0] if tweet_list else "")
        
        print("✅ DrafterAgent finished drafting.")
        return {
            "tweets": tweet_list,
            "selected_hook": selected_hook
        }