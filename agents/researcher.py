from agno.agent import Agent
import sys
import os
import dspy
import threading

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Import the new signature and the config
from dspy_module import GenerateSearchQuery, configure_llm

class ResearchAgent(Agent):
    """
    An agent that formulates a SMART search query from a verbose topic.
    """
    
    def __init__(self):
        super().__init__(
            name="ResearchAgent",
            description="Finds generative AI search results for a topic.",
        )
        # Configure the LLM for this agent
        configure_llm()
        self.query_generator = dspy.Predict(GenerateSearchQuery)
        
        # --- SerpApi Key Rotation Setup ---
        self.serp_keys = []
        # 1. Primary key
        primary = os.getenv("SERPAPI_API_KEY")
        if primary: self.serp_keys.append(primary)
        
        # 2. Numbered keys
        i = 2
        while True:
            k = os.getenv(f"SERPAPI_API_KEY_{i}")
            if not k: break
            self.serp_keys.append(k)
            i += 1
            
        if not self.serp_keys:
            print("⚠️ Warning: No SERPAPI_API_KEY found.")
        else:
            print(f"✅ ResearchAgent loaded {len(self.serp_keys)} SerpApi keys.")
            
        self.current_key_index = 0
        self.key_lock = threading.Lock() # Thread lock for SerpApi keys

    def get_next_serp_key(self):
        """Thread-safe key rotator for SerpApi."""
        if not self.serp_keys: return None
        
        with self.key_lock:
             # Simple Round-Robin
             key = self.serp_keys[self.current_key_index]
             # Move index for next user
             self.current_key_index = (self.current_key_index + 1) % len(self.serp_keys)
             
        masked = key[:4] + "..." + key[-4:]
        print(f"🔎 Using SerpApi Key #{self.current_key_index + 1} ({masked})") 
        return key

    def act(self, topic: str, template: str): 
        """
        Distills a verbose topic into a keyword-rich search query.
        """
        # Rotate key for the LLM itself (just in case)
        configure_llm()
        
        print(f"🔬 ResearchAgent is distilling query from: '{topic[:100]}...'")
        
        # Use the LLM to generate a smart query
        result = self.query_generator(raw_topic=topic)
        query = result.search_query
        
        print(f"✅ ResearchAgent distilled query: '{query}'")
        
        # Get a fresh SerpApi key for this specific request
        api_key = self.get_next_serp_key()
        
        return {"query": query, "api_key": api_key}