# agents/editor.py

from agno.agent import Agent
import sys
import os
import dspy
import ast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.model_loader import ModelLoader
from dspy_module import EditThread, configure_llm
from utils.parsing import safe_parse_list

class EditorAgent(Agent):
    """
    An agent that reviews and refines a drafted thread for quality and style.
    """
    
    def __init__(self):
        super().__init__(
            name="EditorAgent",
            description="Edits a drafted Twitter thread to improve clarity, brevity, and engagement."
        )
        configure_llm()
        self.agentic_model = ModelLoader.get_model()

    # --- THIS IS THE FIX ---
    # We make 'edit_prompt' optional by setting a default value of None
    def act(self, drafted_tweets: list, edit_prompt: str = None):
        """
        Takes a list of tweets and an optional edit prompt.
        Returns an edited version.
        """
        # --- INTELLIGENT ANALYSIS LOGIC ---
        if edit_prompt:
            print(f"🧠 Analyzing user feedback: '{edit_prompt}'...")
            try:
                # Convert user complaint to technical instructions
                analysis = self.agentic_model.analyst_module(
                    draft_thread=str(drafted_tweets), 
                    user_complaint=edit_prompt
                )
                prompt_for_llm = analysis.technical_instructions
                print(f"   -> Translated to: '{prompt_for_llm}'")
            except Exception as e:
                print(f"⚠️ Analysis failed ({e}), using raw prompt.")
                prompt_for_llm = edit_prompt
        else:
            prompt_for_llm = "Review and edit. Make it sound human, remove hashtags, use short paragraphs."
            print(f"🖋️ EditorAgent is doing a general review...")
        # --- END NEW LOGIC ---

        # Convert list to string for the DSPy module input
        draft_str = str(drafted_tweets)
        
        # Pass the determined prompt to the module
        # Pass the determined prompt to the module
        # Note: We need to ensure editor_module exists in AgenticThreadCreator or use a direct Predict here.
        # For now, let's assume we added it or use a fresh Predict if not. 
        # Actually, let's use a direct Predict but configured with the global LM.
        # But wait, to be consistent with loading, we should have added it to AgenticThreadCreator.
        # I will do that in a separate step. For now let's assume it exists.
        result = self.agentic_model.editor_module(draft_thread=draft_str, edit_prompt=prompt_for_llm)
        
        # Safely parse the output back into a list
        # Safely parse the output back into a list
        edited_list = safe_parse_list(result.edited_thread, fix_with_llm=True)

        if not edited_list:
            print("⚠️ Warning: Could not parse edited thread, returning raw string.")
            edited_list = [result.edited_thread]

        print("✅ EditorAgent finished.")
        return {"edited_tweets": edited_list}

    def refine_full_post(self, hook: str, tweets: list, cta: str, prompt: str):
        """
        Refines the entire post (Hook + Tweets + CTA) as a single unit.
        """
        # No extra config needed, agentic_model is loaded
        print(f"🖌️ EditorAgent is refining FULL post with prompt: '{prompt}'")
        
        try:
            result = self.agentic_model.refine_full_module(
                current_hook=hook,
                current_body=str(tweets),
                current_cta=cta,
                instruction=prompt
            )
            
            # Parse body
            refined_body = safe_parse_list(result.refined_body, fix_with_llm=True)
            
            return {
                "refined_hook": result.refined_hook,
                "refined_tweets": refined_body,
                "refined_cta": result.refined_cta
            }
        except Exception as e:
            print(f"⚠️ Full Refinement Failed: {e}")
            return {"error": str(e)}